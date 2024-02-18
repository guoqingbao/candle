use anyhow::Result;
use candle_core::{test_device, test_utils, Device, IndexOp, Tensor};

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5))
w = torch.randn((2, 4, 3))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv1d(t, w)
print(res.flatten())
res = torch.nn.functional.conv1d(t, w, padding=1)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose1d(t, w_t)
print(res.shape)
print(res)
*/
fn conv1d(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
            1.8025, -0.1536, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278, -1.0124, 0.5599,
        ],
        dev,
    )?
    .reshape((1, 4, 5))?;
    let w = Tensor::new(
        &[
            -0.8404f32, -0.3490, 0.0130, 1.3123, 0.1763, -1.9249, 1.4270, 0.9421, 0.8670, -0.7181,
            -1.1111, 0.8869, -1.2429, 1.8357, 1.6052, -1.3844, 0.3951, -1.2036, 0.6686, 1.6261,
            -0.6451, -0.0840, -1.4247, 0.5512,
        ],
        dev,
    )?
    .reshape((2, 4, 3))?;
    let res = t.conv1d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [2.6357, -1.3336, 4.1393, -1.1784, 3.5675, 0.5069]
    );
    let res = t.conv1d(&w, /*padding*/ 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 5]);
    // Same as pytorch default padding: use zeros.
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [2.4509, 2.6357, -1.3336, 4.1393, 0.5657, 1.8091, -1.1784, 3.5675, 0.5069, 3.3352]
    );
    let res = t.conv_transpose1d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 7]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            0.0699, -1.2899, 8.3018, 5.5873, 2.4572, -2.6143, -0.0706, 1.8765, 4.8318, 1.1538,
            4.7076, -5.9745, -0.8276, 1.621
        ],
    );
    Ok(())
}

fn conv1d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[0.4056f32, -0.8689, -0.0773, -1.5630], dev)?.reshape((1, 1, 4))?;
    let w = Tensor::new(&[1f32, 0., 0.], dev)?.reshape((1, 1, 3))?;
    let res = t.conv1d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.4056, -0.8689]
    );
    let res = t.conv1d(&w, /*padding*/ 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.0, 0.4056, -0.8689, -0.0773],
    );
    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5, 5))
w = torch.randn((2, 4, 3, 3))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t, w_t)
print(res.shape)
print(res)

res = torch.nn.functional.conv2d(t, w, dilation=2)
print(res.shape)
print(res[0])

res = torch.nn.functional.conv_transpose2d(t, w_t, dilation=2)
print(res.shape)
print(res)
*/
fn conv2d(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130, 1.3123,
            1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071, 1.1586,
            0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090, 0.2049,
            0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323, -1.3712,
            0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742, 0.3790,
            -0.4431, -0.4720, -0.7890, 0.2620, 0.7875, 0.5377, -0.6779, -0.8088, 1.9098, 1.2006,
            -0.8000, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
        ],
        dev,
    )?;
    let w = Tensor::new(
        &[
            -0.9325f32, 0.6451, -0.8537, 0.2378, 0.8764, -0.1832, 0.2987, -0.6488, -0.2273,
            -2.4184, -0.1192, -0.4821, -0.5079, -0.5766, -2.4729, 1.6734, 0.4558, 0.2851, 1.1514,
            -0.9013, 1.0662, -0.1817, -0.0259, 0.1709, 0.5367, 0.7513, 0.8086, -2.2586, -0.5027,
            0.9141, -1.3086, -1.3343, -1.5669, -0.1657, 0.7958, 0.1432, 0.3896, -0.4501, 0.1667,
            0.0714, -0.0952, 1.2970, -0.1674, -0.3178, 1.0677, 0.3060, 0.7080, 0.1914, 1.1679,
            -0.3602, 1.9265, -1.8626, -0.5112, -0.0982, 0.2621, 0.6565, 0.5908, 1.0089, -0.1646,
            1.8032, -0.6286, 0.2016, -0.3370, 1.2555, 0.8009, -0.6488, -0.4652, -1.5685, 1.5860,
            0.5583, 0.4623, 0.6026,
        ],
        dev,
    )?;
    let t = t.reshape((1, 4, 5, 5))?;
    let w = w.reshape((2, 4, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            -4.2812, 2.0923, 5.2187, 7.5184, 0.752, -14.9426, 10.0087, 4.391, 0.2918, 1.6715,
            10.389, 3.6023, -4.2808, 0.2672, 5.3646, -5.2023, -2.1955, -9.4075
        ]
    );
    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 7, 7]);
    assert_eq!(
        test_utils::to_vec3_round(&res.i(0)?, 4)?,
        [
            [
                [-1.9918, 2.6797, -0.4599, -1.6037, 1.4131, -2.4012, 2.9277],
                [1.8016, -3.5361, 1.0757, 3.5395, -8.2168, -3.2023, 0.5375],
                [0.8243, 1.8675, 7.8929, -4.0746, -6.4415, 5.1139, 1.6889],
                [0.2722, 8.9679, 3.3477, 1.8514, -4.2896, -3.8228, -7.5632],
                [-8.5412, -5.8142, -7.1587, -1.6095, 0.4651, 0.2748, -2.0985],
                [2.0833, -0.6482, -12.1692, -4.1284, -2.9765, -0.0656, -4.5114],
                [5.307, 2.6957, 2.3087, 1.0478, 0.7808, -1.1519, -0.9579]
            ],
            [
                [1.089, 0.1872, -0.6408, -0.9897, 0.8503, 1.1019, -0.9211],
                [-0.1741, -0.2915, 4.2472, 1.9417, 1.65, 0.6303, -4.7131],
                [1.6555, 2.4026, -2.9293, 2.9953, 0.5328, 3.5873, -0.9621],
                [-1.4289, -3.2787, 4.1747, -6.0341, -4.6341, -5.7945, 4.142],
                [7.5973, 6.4431, 5.9872, 2.1639, -8.6566, 3.3143, -3.4059],
                [-0.8775, -3.048, 11.6543, 0.6442, 2.3218, -0.4765, 1.1516],
                [-5.5423, -2.5188, 1.0754, -0.0563, -2.9386, -1.1504, 1.0171]
            ]
        ]
    );
    // Dilations.
    let res = t.conv2d(&w, 0, 1, 2, 1)?;
    assert_eq!(res.dims(), [1, 2, 1, 1]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [2.45, -2.3504],
    );

    // Transpose and dilations.
    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 2)?;
    assert_eq!(res.dims(), [1, 2, 9, 9]);
    assert_eq!(
        test_utils::to_vec3_round(&res.i(0)?, 4)?,
        [
            [
                [-1.9918, 3.1652, -0.6778, -4.3442, 4.4351, 0.6652, -3.0124, -0.6031, 2.9277],
                [2.7036, -1.7156, -0.3969, 1.0516, 1.6381, -2.8886, -0.205, 2.4682, -1.0499],
                [-0.9459, 3.1631, 3.707, -4.8369, -8.5166, -1.4496, -2.7559, -3.2698, 1.4376],
                [-0.2157, 3.7786, -2.0252, -4.2633, 3.6731, -1.5142, 5.9391, -0.2622, -0.141],
                [-6.8121, -3.1744, 1.5945, 3.0637, -9.6088, 1.4446, 2.9489, -3.0082, -7.3822],
                [0.2371, 3.3303, 0.3861, 2.2646, -4.6784, 4.1235, -0.0109, 0.3176, -0.03],
                [-2.5339, -2.9564, -3.4518, -4.4594, -9.1873, -1.9709, -0.4676, 0.51, -3.5024],
                [4.007, 0.3067, -2.2954, 1.1105, -0.1992, 1.6372, -2.9268, 0.2807, -1.2787],
                [5.307, 1.1317, 1.3518, 0.9049, 3.8116, -0.4075, -0.8874, -0.2241, -0.9579]
            ],
            [
                [1.089, -0.6483, 0.0726, -0.4752, -1.3283, 1.7103, 1.0703, 0.1076, -0.9211],
                [-0.8629, 0.1376, 0.3202, 2.0955, 0.9696, 2.8988, -1.0012, 1.5049, -0.1278],
                [1.9286, -1.5255, -2.9563, 2.4589, 3.3611, -0.6951, 0.3525, -1.7724, -5.9861],
                [1.1226, 2.1561, 3.6417, 4.7546, -0.692, 4.4126, -5.1902, 6.0805, 2.3185],
                [1.0111, 0.3604, 0.6432, -3.6605, 7.9517, -9.2955, -5.2988, -3.7803, -2.0642],
                [3.3172, -1.7967, -3.6576, -2.0942, 1.3158, 0.112, -1.7405, 2.9167, 0.7957],
                [5.1001, 1.8995, -1.8639, 1.1262, 9.9629, 2.683, -3.6319, -1.1607, 0.5856],
                [-4.8445, -0.5642, 4.2317, 0.0856, 1.2267, -0.5712, 1.736, 1.0997, 0.6908],
                [-5.5423, -1.1831, -1.2176, 0.0843, 0.0446, -0.7545, -2.4798, -0.0827, 1.0171]
            ]
        ]
    );
    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 2, 3, 3))
w = torch.randn((1, 2, 1, 1))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t, w_t)
print(res.shape)
print(res.flatten())

t_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t_t, w)
print(res.shape)
print(res.flatten())
*/
fn conv2d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
            -0.6266, 0.3529, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278,
        ],
        dev,
    )?;
    let w = Tensor::new(&[-0.9259f32, 1.3017], dev)?;
    let t = t.reshape((1, 2, 3, 3))?;
    let w = w.reshape((1, 2, 1, 1))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.164, -0.0111, -0.1742, 2.6437, -2.0268, 1.1823, 3.2855, -1.0324, 0.2539]
    );
    let res = t.conv2d(&w, 2, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 7, 7]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1640, -0.0111, -0.1742, 0.0000, 0.0000,
            0.0000, 0.0000, 2.6437, -2.0268, 1.1823, 0.0000, 0.0000, 0.0000, 0.0000, 3.2855,
            -1.0324, 0.2539, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ]
    );
    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.164, -0.0111, -0.1742, 2.6437, -2.0268, 1.1823, 3.2855, -1.0324, 0.2539],
    );
    let res = t.transpose(0, 1)?.conv_transpose2d(&w, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [2, 2, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            -0.3755, 0.8045, -0.6336, -0.2218, -1.1369, 0.8599, 1.5768, -0.1268, -0.1728, 0.528,
            -1.131, 0.8908, 0.3118, 1.5984, -1.2089, -2.2168, 0.1783, 0.2429, -0.3838, 0.5802,
            -0.3268, -2.0382, 0.6329, -0.2293, -1.2154, 0.6441, -0.3035, 0.5396, -0.8156, 0.4594,
            2.8654, -0.8898, 0.3224, 1.7087, -0.9056, 0.4267
        ]
    );
    Ok(())
}

fn conv2d_smaller(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866,
        ],
        dev,
    )?;
    let w = Tensor::new(&[1f32, 1., 1., 1., 1., 1., 1., 1., 1.], dev)?;
    let t = t.reshape((1, 1, 3, 3))?;
    let w = w.reshape((1, 1, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 1, 1]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [-0.6197]
    );
    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 2, 4, 2))
w = torch.randn((1, 2, 1, 1))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())
*/
fn conv2d_non_square(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699,
        ],
        dev,
    )?;
    let w = Tensor::new(&[-1.1351f32, 1.3841], dev)?;
    let t = t.reshape((1, 2, 4, 2))?;
    let w = w.reshape((1, 2, 1, 1))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.2312, 5.2238, 2.3772, 1.9076, 2.0256, -0.5776, -1.6028, -1.467]
    );
    Ok(())
}

/*
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5, 5), requires_grad=True)
w = torch.randn((2, 4, 3, 3), requires_grad=True)
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())
loss = (res ** 2).sum()
print(loss)
loss.backward()
print(t.grad.shape)
print(t.grad.flatten())
print(w.grad.shape)
print(w.grad.flatten())

t.grad.zero_()
w.grad.zero_()
res = torch.nn.functional.conv2d(t, w, stride=2)
print(res.flatten())
loss = (res ** 2).sum()
print(loss)
loss.backward()
print(t.grad.shape)
print(t.grad[0])
print(w.grad.shape)
print(w.grad[0])
*/
fn conv2d_grad(dev: &Device) -> Result<()> {
    use candle_core::Var;
    let t = Var::from_slice(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130, 1.3123,
            1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071, 1.1586,
            0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090, 0.2049,
            0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323, -1.3712,
            0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742, 0.3790,
            -0.4431, -0.4720, -0.7890, 0.2620, 0.7875, 0.5377, -0.6779, -0.8088, 1.9098, 1.2006,
            -0.8000, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
        ],
        (1, 4, 5, 5),
        dev,
    )?;
    let w = Var::from_slice(
        &[
            -0.9325f32, 0.6451, -0.8537, 0.2378, 0.8764, -0.1832, 0.2987, -0.6488, -0.2273,
            -2.4184, -0.1192, -0.4821, -0.5079, -0.5766, -2.4729, 1.6734, 0.4558, 0.2851, 1.1514,
            -0.9013, 1.0662, -0.1817, -0.0259, 0.1709, 0.5367, 0.7513, 0.8086, -2.2586, -0.5027,
            0.9141, -1.3086, -1.3343, -1.5669, -0.1657, 0.7958, 0.1432, 0.3896, -0.4501, 0.1667,
            0.0714, -0.0952, 1.2970, -0.1674, -0.3178, 1.0677, 0.3060, 0.7080, 0.1914, 1.1679,
            -0.3602, 1.9265, -1.8626, -0.5112, -0.0982, 0.2621, 0.6565, 0.5908, 1.0089, -0.1646,
            1.8032, -0.6286, 0.2016, -0.3370, 1.2555, 0.8009, -0.6488, -0.4652, -1.5685, 1.5860,
            0.5583, 0.4623, 0.6026,
        ],
        (2, 4, 3, 3),
        dev,
    )?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2)?, 741.12f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&grad_t.flatten_all()?, 2)?,
        [
            9.29, -2.84, -5.71, 3.38, -7.71, -19.15, 7.02, 29.1, 9.34, 34.73, -22.87, 24.35,
            -39.88, -14.01, 21.08, 9.94, 13.63, -34.68, 11.21, -6.26, 7.72, -6.32, -16.64, -1.08,
            -20.22, 21.73, -0.37, -4.06, 5.82, -3.65, -30.73, 14.55, 87.7, 31.6, 4.53, -89.78,
            -75.37, -57.43, -7.56, 92.96, 18.79, -4.63, -159.75, -42.47, -47.26, 52.88, 37.32,
            49.0, 12.82, 2.01, -8.98, 20.18, 16.62, 12.06, 15.38, 20.0, 2.57, -15.22, 72.62,
            -10.75, 2.25, -31.2, 3.75, -0.2, 9.76, -0.68, 5.21, -40.44, -22.59, -61.61, 17.28,
            20.41, 37.55, 5.23, 6.81, 23.54, 23.62, -9.99, -9.13, 4.87, -35.06, -26.1, 63.48,
            25.81, -39.21, -70.68, -46.96, 2.33, 41.81, 82.42, -28.63, -11.78, -35.33, -10.28,
            -28.57, -9.13, 7.21, -9.05, -9.62, -11.25
        ]
    );
    assert_eq!(
        test_utils::to_vec1_round(&grad_w.flatten_all()?, 2)?,
        [
            -28.92, -22.88, -141.23, 73.35, 61.07, 47.81, -20.0, -73.71, -41.82, -13.59, 21.5,
            28.72, 28.57, -46.85, -90.19, 143.61, 16.68, 7.43, 18.88, -90.81, -20.29, 54.79, 82.63,
            22.94, 77.81, -16.39, -13.2, 9.34, -40.39, -26.62, 5.33, -60.91, 9.09, -59.37, 7.08,
            58.64, 5.55, 20.52, 2.5, -17.25, -6.8, 22.21, 30.15, -7.52, -37.46, 5.67, 22.58, 9.03,
            47.05, 17.61, 37.31, -98.13, -14.61, -4.8, -6.36, 44.69, 23.34, 8.37, -13.52, 80.05,
            -34.24, -16.36, -12.31, 1.92, -33.62, -14.1, -49.23, -7.39, 11.5, -9.98, 9.66, 29.6
        ]
    );

    // Same as before but with stride.
    let res = t.conv2d(&w, 0, 2, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2)?, 277.16f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        test_utils::to_vec3_round(&grad_t.i(0)?, 2)?,
        [
            [
                [9.29, -7.03, 0.94, 3.49, -7.71],
                [-1.8, -7.82, 8.9, 8.46, 7.43],
                [-25.84, 22.09, -19.27, -0.22, 1.69],
                [4.02, 18.53, -18.37, 2.3, -24.51],
                [7.72, -9.68, -12.34, 5.6, -20.22]
            ],
            [
                [21.73, 3.39, -18.27, 3.86, -3.65],
                [8.25, 3.73, 30.73, -8.61, -11.93],
                [-72.15, -15.36, -17.53, -12.32, -1.61],
                [-22.32, -7.79, -91.82, 6.44, -37.69],
                [52.88, 14.44, 42.75, 9.88, 2.01]
            ],
            [
                [-8.98, 9.91, 6.75, -4.68, 15.38],
                [4.93, -0.33, 9.94, -1.46, 14.78],
                [13.62, -30.63, 3.96, -3.58, -4.48],
                [-14.13, 1.19, -34.43, 3.08, -33.83],
                [17.28, 12.94, 31.83, -3.35, 6.81]
            ],
            [
                [23.54, 6.98, -24.52, 0.52, 4.87],
                [9.65, 6.18, 1.71, -25.23, -4.93],
                [-54.99, -23.66, 3.19, -3.73, 18.58],
                [-21.35, -10.39, -39.88, 28.73, -30.76],
                [-9.13, 11.12, -14.0, -8.23, -11.25]
            ]
        ]
    );
    assert_eq!(
        test_utils::to_vec3_round(&grad_w.i(0)?, 2)?,
        [
            [
                [28.34, -7.91, -45.75],
                [21.03, 3.86, 29.86],
                [0.72, -36.58, -35.28]
            ],
            [
                [-16.04, 11.53, -16.38],
                [29.62, -16.32, -48.35],
                [57.5, 28.29, 25.81]
            ],
            [
                [2.93, -19.6, 1.57],
                [27.15, 53.88, -24.64],
                [12.74, -22.6, -26.2]
            ],
            [
                [-0.18, -14.86, -6.82],
                [-19.55, -2.72, 45.9],
                [-2.54, 36.97, 27.11]
            ]
        ]
    );

    // Replicate the issue from https://github.com/huggingface/candle/issues/1212
    let res = t.i((.., .., 0..4, 0..4))?.conv2d(&w, 0, 2, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2)?, 21.12f32);
    let grads = loss.backward()?;
    let grad_t = grads.get(&t).unwrap();
    let grad_w = grads.get(&w).unwrap();
    assert_eq!(grad_t.dims(), [1, 4, 5, 5]);
    assert_eq!(grad_w.dims(), [2, 4, 3, 3]);
    assert_eq!(
        test_utils::to_vec3_round(&grad_t.i(0)?, 2)?,
        [
            [
                [9.29, -7.03, 7.87, 0.0, 0.0],
                [-1.8, -7.82, 5.9, 0.0, 0.0],
                [-3.12, 4.49, 5.52, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [21.73, 3.39, 4.77, 0.0, 0.0],
                [8.25, 3.73, 27.61, 0.0, 0.0],
                [-20.55, -5.61, -2.77, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [-8.98, 9.91, -7.15, 0.0, 0.0],
                [4.93, -0.33, 4.56, 0.0, 0.0],
                [-6.7, -5.76, -8.05, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            [
                [23.54, 6.98, -10.0, 0.0, 0.0],
                [9.65, 6.18, 18.72, 0.0, 0.0],
                [3.29, -5.27, 0.79, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ]
    );
    assert_eq!(
        test_utils::to_vec3_round(&grad_w.i(0)?, 2)?,
        [
            [
                [-3.47, 7.44, 0.66],
                [12.89, -3.4, -9.29],
                [-14.16, -0.83, 7.14]
            ],
            [
                [-3.23, 5.37, -3.02],
                [-2.12, -11.24, 1.94],
                [6.97, 7.2, 2.99]
            ],
            [
                [-4.04, -3.31, 4.87],
                [-6.68, -5.68, 1.73],
                [-5.54, 4.32, 0.52]
            ],
            [[-4.72, 1.5, 4.72], [3.79, 4.04, 6.76], [-4.6, 5.8, 6.93]]
        ]
    );

    Ok(())
}

test_device!(conv1d, conv1d_cpu, conv1d_gpu, conv1d_metal);
test_device!(
    conv1d_small,
    conv1d_small_cpu,
    conv1d_small_gpu,
    conv1d_small_metal
);
test_device!(conv2d, conv2d_cpu, conv2d_gpu, conv2d_metal);
test_device!(
    conv2d_non_square,
    conv2d_non_square_cpu,
    conv2d_non_square_gpu,
    conv2d_non_square_metal
);
test_device!(
    conv2d_small,
    conv2d_small_cpu,
    conv2d_small_gpu,
    conv2d_small_metal
);
test_device!(
    conv2d_smaller,
    conv2d_smaller_cpu,
    conv2d_smaller_gpu,
    conv2d_smaller_metal
);
test_device!(
    conv2d_grad,
    conv2d_grad_cpu,
    conv2d_grad_gpu,
    conv2_grad_metal
);
