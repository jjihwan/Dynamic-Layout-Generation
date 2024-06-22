from einops import repeat

def make_dynamic(label, bbox, mask, num_frame=4, type="flip"):
    label = repeat(label, 'b n -> b t n', t=num_frame)
    bbox = repeat(bbox, 'b n d-> b t n d', t=num_frame).clone()
    mask = repeat(mask, 'b n -> b t n', t=num_frame)

    if type == "flip":
        bbox[:, 1, :, 0] = 1 - bbox[:, 0, :, 0]
        bbox[:, 2, :, :2] = 1 - bbox[:, 0, :, :2]
        bbox[:, 3, :, 1] = 1 - bbox[:, 0, :, 1]
        bbox = bbox * mask.unsqueeze(-1)
    elif type == "x_shift":
        bbox[:, 0, :, 0] = bbox[:, 0, :, 0] - 0.04
        bbox[:, 1, :, 0] = bbox[:, 1, :, 0] - 0.02
        bbox[:, 2, :, 0] = bbox[:, 2, :, 0] + 0.02
        bbox[:, 3, :, 0] = bbox[:, 3, :, 0] + 0.04
        bbox = bbox * mask.unsqueeze(-1)
    elif type == "y_shift":
        bbox[:, 0, :, 1] = bbox[:, 0, :, 1] - 0.06
        bbox[:, 1, :, 1] = bbox[:, 1, :, 1] - 0.03
        bbox[:, 2, :, 1] = bbox[:, 2, :, 1] + 0.03
        bbox[:, 3, :, 1] = bbox[:, 3, :, 1] + 0.06
        bbox = bbox * mask.unsqueeze(-1)
    elif type == "rotate":
        bbox[:, 0, :, 0] = bbox[:, 0, :, 0] - 0.06
        bbox[:, 0, :, 1] = bbox[:, 0, :, 1] - 0.06
        bbox[:, 1, :, 0] = bbox[:, 1, :, 0] + 0.06
        bbox[:, 1, :, 1] = bbox[:, 1, :, 1] - 0.06
        bbox[:, 2, :, 0] = bbox[:, 2, :, 0] + 0.06
        bbox[:, 2, :, 1] = bbox[:, 2, :, 1] + 0.06
        bbox[:, 3, :, 0] = bbox[:, 3, :, 0] - 0.06
        bbox[:, 3, :, 1] = bbox[:, 3, :, 1] + 0.06
        bbox = bbox * mask.unsqueeze(-1)
    elif type == "static":
        bbox = bbox * mask.unsqueeze(-1)
    else:
        raise ValueError("Unknown type")

    return label, bbox, mask