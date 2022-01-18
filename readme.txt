1. Train Spatial network

model = Networks.get_net_only_spatial(batch_size, patch_shape[0], patch_shape[1],patch_shape[2],patch_shape[3])

2. Train Temporal network

model = Networks_finetuning.get_net_border(batch_size, patch_shape[0], patch_shape[1],patch_shape[2],patch_shape[3], None, False)

3. Fine-Tuning

model = Networks_finetuning.get_net_border(batch_size, patch_shape[0], patch_shape[1],patch_shape[2],patch_shape[3],spatial_model_pretrained, True)

NOTE: spatial_model_pretrained is the trained weight in step 2