# nii-rs 重构记录

## 背景

原有代码依赖 `nifti-rs` + `nalgebra`，两套版本不兼容（nalgebra 0.32 vs 0.33），导致编译错误。
此外 `ijk2xyz`/`xyz2ijk` 的实现存在逻辑错误（忽略 direction），Python 绑定使用宏生成 10 个类，架构臃肿。

## 整体策略

1. 自写 NIfTI-1 header 解析器，彻底删除 `nifti-rs` 和 `nalgebra`
2. 仿射矩阵全部用 `ndarray::Array2<f64>` 存储，3×3 逆矩阵用解析公式
3. Python 绑定改为单一 `#[pyclass]` + 内部枚举
4. 以 nibabel / SimpleITK 为金标准验证正确性

---

## 第一轮：核心解析器

### 文件结构变化

```
之前                          之后
src/
├── lib.rs                    src/
├── image.rs  (nifti-rs 代理)   ├── lib.rs
├── utils.rs (nalgebra 桥接)    ├── header.rs  ← 新增
└── bind.rs  (宏爆炸)           ├── image.rs   ← 重写
                               └── bind.rs    ← 重写
```

### `header.rs` — NIfTI-1 348 字节解析

#### 字节序检测

sizeof_hdr (偏移 0) 必须等于 348。分别按 LE/BE 读取，匹配的即为文件字节序：

```rust
let le_size = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
let be_size = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
let little_endian = if le_size == 348 { true }
    else if be_size == 348 { false }
    else { return Err(...) };
```

关键：`i32::from_le_bytes` 和 `i32::from_be_bytes` 是 Rust 标准库函数，不依赖任何第三方 crate。

#### 正确字段偏移

⚠️ **最容易踩的坑**：NIfTI-1 的 C 结构体有对齐填充。字段实际偏移如下（按 nibabel/nifti1.h 标准）：

| 字段 | 偏移 | 大小 | 说明 |
|------|------|------|------|
| sizeof_hdr | 0 | 4 | 必须为 348 |
| dim[8] | 40 | 16 | dim[0]=维度数，dim[1..]=各维大小 |
| datatype | 70 | 2 | 16=f32, 4=i16, 2=u8 ... |
| bitpix | 72 | 2 | 每像素位数 |
| pixdim[8] | 76 | 32 | pixdim[0]=qfac, pixdim[1..]=spacing |
| vox_offset | 108 | 4 | 数据起始偏移 |
| qform_code | 252 | 2 | 四元数形式编码 |
| sform_code | 254 | 2 | 仿射形式编码 |
| quatern_b/c/d | 256-267 | 12 | 四元数参数 |
| qoffset_x/y/z | 268-279 | 12 | 四元数偏移 |
| srow_x/y/z | 280-327 | 48 | 仿射矩阵行 |
| magic | 344 | 4 | "n+1\0" 或 "ni1\0" |

所有字段定义在文件顶部为命名常量，一目了然：

```rust
pub(crate) const OFF_DIM: usize = 40;
pub(crate) const OFF_DATATYPE: usize = 70;
// ...
```

> **血泪教训**：最初我用了偏移 24 做 dim、偏移 54 做 datatype……全部差 16 字节，导致解析完全错误。因为 `dim` 前面有 `sizeof_hdr(4) + data_type(10) + db_name(18) + extents(4) + session_error(2) + regular(1) + dim_info(1) = 40`。

#### 仿射矩阵优先级

NIfTI-1 标准定义了三层仿射：

1. **sform**（优先）：直接使用 `srow_x/y/z` 构造 4×4 矩阵
2. **qform**（备选）：由四元数 `(b,c,d)` + `pixdim` 计算旋转矩阵
3. **pixdim 对角**（兜底）：仅对角线有值，origin=0

```rust
pub fn affine(&self) -> Array2<f64> {
    if self.sform_code > 0 { self.affine_from_sform() }
    else if self.qform_code > 0 { self.affine_from_qform() }
    else { self.affine_from_pixdim() }
}
```

四元数转旋转矩阵公式（NIfTI-1 标准）：

```
a = sqrt(1 - b² - c² - d²)

R[0][0] = a² + b² - c² - d²
R[0][1] = 2bc - 2ad
R[0][2] = 2bd + 2ac
R[1][0] = 2bc + 2ad
R[1][1] = a² + c² - b² - d²
R[1][2] = 2cd - 2ab
R[2][0] = 2bd - 2ac
R[2][1] = 2cd + 2ab
R[2][2] = a² + d² - c² - b²

affine[:3, 0] = R[:3, 0] * pixdim[1]
affine[:3, 1] = R[:3, 1] * pixdim[2]
affine[:3, 2] = R[:3, 2] * pixdim[3] * qfac
affine[:3, 3] = [qoffset_x, qoffset_y, qoffset_z]
```

---

### `image.rs` — Nifti1Image 结构体

#### 数据类型绑定

不再需要 `DataElement` trait，自定 `NiftiType` trait：

```rust
pub trait NiftiType: Pod + Send + Sync + 'static {
    const DATATYPE: i16;
    const BITPIX: i16;
}

impl NiftiType for f32 { const DATATYPE: i16 = 16; const BITPIX: i16 = 32; }
impl NiftiType for u8  { const DATATYPE: i16 = 2;  const BITPIX: i16 = 8;  }
// ...
```

#### 数据排列约定

| 系统 | 数据排列 | 坐标系统 |
|------|---------|---------|
| **nifti-rs / nibabel** | `[x, y, z]` | RAS+ |
| **SimpleITK / nii-rs** | `[z, y, x]` | LPS |
| **NumPy (nii-rs)** | `[z, y, x]` | LPS |

数据读取流程：

```
文件 [x,y,z] bytes ──→ 重排为 [z,y,x] ──→ ndarray
                                                 ↓
ndarray ──→ 重排为 [x,y,z] ──→ 文件 bytes
```

关键：数据重排是**显式的三重循环**，而非 `permuted_axes`。因为 `permuted_axes` 返回的数组在 ndarray 中的行为与我们预期的不同——它创建一个具有不同 strides 的视图，而不是一个 C-contiguous 重排数组。

```rust
// 读取: [x,y,z] → [z,y,x]
let mut itk = vec![0u8; n * elem_sz];
for z in 0..nz {
    for y in 0..ny {
        for x in 0..nx {
            let src = (x + nx*y + nx*ny*z) * elem_sz;
            let dst = (z*ny*nx + y*nx + x) * elem_sz;
            itk[dst..dst+elem_sz].copy_from_slice(&raw[src..src+elem_sz]);
        }
    }
}

// 写入: [z,y,x] → [x,y,z]
for z in 0..nz {
    for y in 0..ny {
        for x in 0..nx {
            let dst = (x + nx*y + nx*ny*z) * elem_sz;
            // 从 arr[[z,y,x]] 读取，写入 file_data[dst]
        }
    }
}
```

#### 3×3 矩阵求逆（无 nalgebra）

用解析公式，纯 ndarray：

```
det = a(ei−fh) − b(di−fg) + c(dh−eg)

R⁻¹ = 1/det × [
    (ei−fh),  (ch−bi),  (bf−ce),
    (fg−di),  (ai−cg),  (cd−af),
    (dh−eg),  (bg−ah),  (ae−bd),
]
```

4×4 仿射求逆：

```
A = [[R t],    A⁻¹ = [[R⁻¹  -R⁻¹t],
     [0 1]]           [  0      1  ]]
```

#### ijk2xyz / xyz2ijk 的正确实现

**之前**（有 bug）：只用了 spacing × index + origin，完全忽略 direction。

**之后**：使用完整仿射矩阵，包含 RAS↔LPS 转换：

```
ijk2xyz([i,j,k]):
  ─→ nifti index [x,y,z] = [k,j,i]            (ITK → nifti)
  ─→ RAS = affine × [x,y,z,1]                   (nifti 索引 → RAS)
  ─→ LPS = [-RAS_x, -RAS_y, RAS_z]               (RAS → LPS)

xyz2ijk([x,y,z]):
  ─→ RAS = [-x, -y, z]                           (LPS → RAS)
  ─→ nifti index = R⁻¹ × (RAS_xyz - t)           (逆仿射)
  ─→ ITK [z,y,x] = [nifti_z, nifti_y, nifti_x]  (nifti → ITK)
```

---

### 依赖精简

```
之前:                         之后:
nifti     (繁重)              ndarray    (核心数据结构)
nalgebra  (版本冲突)           bytemuck   (安全类型转换)
simba     (无人用)             flate2     (gzip)
num-traits(无人用)             rayon      (并行)
itertools (无人用)
ndarray-ndimage (无人用)

~10 个 → 4 个核心依赖
```

在 `Cargo.toml` 中 `pyo3` 和 `numpy` 改为可选 feature，不编译 Python 绑定时不依赖它们：

```toml
[features]
default = []
python = ["pyo3", "numpy"]
```

---

## 第二轮：Python 绑定

### 之前的问题

宏生成 10 个 Python 类 + 40 个 Python 函数：

```
Rust 宏展开:                     Python dispatch:
  impl_py_wrapper!(f32)  ──→     _nii.read_image_f32()
  impl_py_wrapper!(u8)   ──→     _nii.read_image_u8()
  ... × 10                      ... × 10
                                ↓ 还要 __init__.py 里写
                                funcs = { np.float32: _nii.read_image_f32, ... }
```

任何改动需要同时修改 Rust 宏 + Python dispatch dict。

### 现在的方案

一个 `#[pyclass]` + `ImageData` 内部枚举：

```rust
enum ImageData {
    F32(Array3<f32>),
    F64(Array3<f64>),
    U8(Array3<u8>),
    // ... 10 种类型
}

#[pyclass(name = "Nifti1Image")]
struct PyNifti1Image {
    header: Nifti1Header,
    data: ImageData,
}
```

`__init__.py` 从 240 行减到 14 行：

```python
from nii._nii import Nifti1Image
__all__ = ["Nifti1Image"]
read_image = Nifti1Image.read
```

### 关键实现细节

**读取文件**：`read(path)` 是静态方法，自动检测 dtype：

```rust
#[staticmethod]
fn read(path: &str) -> PyResult<Self> {
    let bytes = read_file_bytes(Path::new(path))?;
    let header = Nifti1Header::parse(&bytes)?;
    let data = read_data(&bytes, &header)?;  // 自动匹配 dtype
    Ok(PyNifti1Image { header, data })
}
```

**ndarray() 返回**：通过 `IntoPyArray` + `into_any().into()` 转换为 `PyObject`：

```rust
fn ndarray<'py>(&self, py: Python<'py>) -> PyObject {
    match &self.data {
        ImageData::F32(a) => a.clone().into_pyarray(py).into_any().into(),
        // ...
    }
}
```

**`new(arr, affine)`**：接受任意 numpy ndarray，检测 dtype 后构造对应变体：

```rust
#[staticmethod]
fn new(arr: &Bound<'py, PyAny>, affine: PyReadonlyArray2<f64>) -> PyResult<Self> {
    let dtype_s = arr.getattr("dtype")?.str()?.to_string();
    let bytes = arr.call_method0("tobytes")?.extract::<Vec<u8>>()?;
    // ... match dtype_s ...
}
```

**错误处理**：实现 `From<NiftiError> for PyErr`，Rust 错误自动转为 Python 异常。

---

## 第三轮：测试体系

### Rust 单元测试（7 个）

```
cargo test --lib

test header::tests::test_parse_minimal_header   手工构造 348 字节验证解析
test header::tests::test_header_big_endian      大端字节序测试
test image::tests::test_read_nifti_from_bytes   从字节流读取并验证数据
test image::tests::test_round_trip              to_bytes → from_bytes 往返
test image::tests::test_affine_round_trip       set_affine → get_affine 往返
test image::tests::test_origin_direction        get_origin/get_direction 验证
test image::tests::test_ijk2xyz_identity        ijk2xyz/xyz2ijk 往返验证
```

### Python 端到端测试（8 个）

```
python tests/test_e2e.py

f32 round-trip      from_array → write .nii → read back
u8 round-trip gz    同上，.nii.gz 压缩格式
affine round-trip   get_affine → set_affine → write → read
spacing/origin/dir  set → get → write → read
new() with affine   传入 ndarray + affine
__str__             字符串表示
copy_information    图像间复制仿射信息
ijk2xyz / xyz2ijk   坐标变换往返
```

### 金标准测试（8 个）— 最重要

```
python tests/test_golden.py

1. nibabel 写 .nii → nii-rs 读         验证数据和仿射
2. SimpleITK 写 .nii.gz → nii-rs 读    验证 spacing/origin/direction
3. nii-rs 写 .nii.gz → nibabel 读      反向验证
4. nii-rs 写 .nii → SimpleITK 读       反向验证
5. ijk2xyz vs nibabel                   坐标变换精确性
6. 旋转方向 vs SimpleITK                 非单位方向测试
7. Big-endian                           字节序支持
8. Spacing/Origin/Direction vs both     三者同时验证
```

测试核心理念：nibabel 和 SimpleITK 是业已成熟的实现，**以它们为金标准**，不自行假设行为。

---

## 关键坐标约定速查

| 概念 | nibabel | SimpleITK | **nii-rs** |
|------|---------|-----------|------------|
| 数据排列 | `[x, y, z]` | `[z, y, x]` | `[z, y, x]` |
| 仿射约定 | 行主序，RAS+ | — | nibabel 一致 |
| 坐标系 | RAS+ | LPS | LPS (get_origin/direction) |
| spacing | `[dx, dy, dz]` | `[dx, dy, dz]` | `[dx, dy, dz]` |
| origin | RAS `[tx, ty, tz]` | LPS `[ox, oy, oz]` | LPS `[ox, oy, oz]` |
| direction | — | LPS 3×3 | LPS 3×3 |
| get_affine | 4×4 RAS row-major | — | nibabel 一致 |

**数据排列互换**：
```
nibabel [x,y,z] ←→ nii-rs [z,y,x]:  arr.transpose(2, 1, 0)
```

**仿射互换**：
```
nibabel 仿射 = nii-rs get_affine()  // 完全一致
```

**RAS ↔ LPS**：
```
LPS = [-RAS_x, -RAS_y, RAS_z]
RAS = [-LPS_x, -LPS_y, LPS_z]
```

---

## 待办 / 后续方向

1. **`.hdr/.img` 分离格式** — 解析 .hdr + .img 配对文件。改动量小，核心解析逻辑复用
2. **2D/4D 支持** — 当前仅支持 3D，NIfTI-1 实际可支持 1-7 维
3. **`nii::new(arr, aff)` → Python** — Rust 侧已有，Python 绑定也已暴露
4. **性能基准** — 与 nibabel/SimpleITK 对比 I/O 速度
