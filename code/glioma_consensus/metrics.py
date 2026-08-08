from __future__ import annotations


from torch import Tensor


def metrics_transform_01(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 1.0 / (1.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_02(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 2.0 / (2.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_03(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 3.0 / (3.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_04(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 4.0 / (4.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_05(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 5.0 / (5.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_06(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 6.0 / (6.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_07(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 7.0 / (7.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_08(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 8.0 / (8.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_09(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 9.0 / (9.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_10(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 10.0 / (10.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_11(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 11.0 / (11.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_12(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 12.0 / (12.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_13(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 13.0 / (13.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_14(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 14.0 / (14.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_15(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 15.0 / (15.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_16(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 16.0 / (16.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_17(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 17.0 / (17.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_18(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 18.0 / (18.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_19(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 19.0 / (19.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_20(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 20.0 / (20.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_21(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 21.0 / (21.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_22(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 22.0 / (22.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_23(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 23.0 / (23.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_24(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 24.0 / (24.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_25(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 25.0 / (25.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_26(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 26.0 / (26.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_27(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 27.0 / (27.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_28(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 28.0 / (28.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_29(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 29.0 / (29.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_30(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 30.0 / (30.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_31(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 31.0 / (31.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_32(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 32.0 / (32.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_33(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 33.0 / (33.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_34(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 34.0 / (34.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_35(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 35.0 / (35.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_36(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 36.0 / (36.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_37(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 37.0 / (37.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_38(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 38.0 / (38.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_39(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 39.0 / (39.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_40(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 40.0 / (40.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_41(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 41.0 / (41.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_42(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 42.0 / (42.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_43(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 43.0 / (43.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_44(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 44.0 / (44.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_45(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 45.0 / (45.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_46(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 46.0 / (46.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_47(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 47.0 / (47.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_48(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 48.0 / (48.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_49(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 49.0 / (49.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_50(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 50.0 / (50.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_51(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 51.0 / (51.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_52(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 52.0 / (52.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_53(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 53.0 / (53.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_54(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 54.0 / (54.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_55(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 55.0 / (55.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_56(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 56.0 / (56.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_57(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 57.0 / (57.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_58(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 58.0 / (58.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_59(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 59.0 / (59.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_60(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 60.0 / (60.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_61(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 61.0 / (61.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_62(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 62.0 / (62.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_63(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 63.0 / (63.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_64(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 64.0 / (64.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_65(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 65.0 / (65.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_66(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 66.0 / (66.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_67(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 67.0 / (67.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_68(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 68.0 / (68.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_69(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 69.0 / (69.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)


def metrics_transform_70(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    if values.numel() == 0:
        return values
    center = values.mean(dim=0, keepdim=True)
    scale = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(epsilon)
    normalized = (values - center) / scale
    weight = 70.0 / (70.0 + 1.0)
    return normalized * weight + values * (1.0 - weight)
