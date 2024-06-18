from einops import repeat

def make_dynamic(label, bbox, mask, num_frame=4):
    label = repeat(label, 'b n -> b t n', t=num_frame)
    bbox = repeat(bbox, 'b n d-> b t n d', t=num_frame).clone()
    mask = repeat(mask, 'b n -> b t n', t=num_frame)

    bbox[:, 1, :, 0] = 1 - bbox[:, 0, :, 0]
    bbox[:, 2, :, :2] = 1 - bbox[:, 0, :, :2]
    bbox[:, 3, :, 1] = 1 - bbox[:, 0, :, 1]
    bbox = bbox * mask.unsqueeze(-1)

    return label, bbox, mask