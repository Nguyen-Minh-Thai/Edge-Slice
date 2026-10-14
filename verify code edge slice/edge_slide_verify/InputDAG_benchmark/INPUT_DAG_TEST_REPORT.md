# BÁO CÁO KIỂM THỬ TẬP MÔ HÌNH INPUT DAG

- **Ngày kiểm thử:** 2026-10-10 18:42:38
- **Tổng số mô hình:** 104
- **Thành công (PASS):** 87
- **Bỏ qua (Chưa trace TorchScript):** 17
- **Lỗi (FAIL):** 0

## Chi tiết kết quả các mô hình đã chạy thành công

| Grade | Kiến trúc | Scale | DAG Nodes | Số Params | Input Shape | Output Shape | Latency (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| grade_1_poorfit | `scalar_elementwise_mlp` | scale_100 | 800 | 27200 | `[1, 16]` | `[1, 16]` | 133.67 ms |
| grade_1_poorfit | `scalar_elementwise_mlp` | scale_12 | 96 | 3264 | `[1, 16]` | `[1, 16]` | 12.82 ms |
| grade_1_poorfit | `scalar_elementwise_mlp` | scale_160 | 1280 | 43520 | `[1, 16]` | `[1, 16]` | 205.81 ms |
| grade_1_poorfit | `scalar_elementwise_mlp` | scale_220 | 1760 | 59840 | `[1, 16]` | `[1, 16]` | 310.34 ms |
| grade_1_poorfit | `scalar_elementwise_mlp` | scale_30 | 240 | 8160 | `[1, 16]` | `[1, 16]` | 36.82 ms |
| grade_1_poorfit | `scalar_elementwise_mlp` | scale_300 | 2400 | 81600 | `[1, 16]` | `[1, 16]` | 470.02 ms |
| grade_1_poorfit | `scalar_elementwise_mlp` | scale_5 | 40 | 1360 | `[1, 16]` | `[1, 16]` | 6.0 ms |
| grade_1_poorfit | `scalar_elementwise_mlp` | scale_60 | 480 | 16320 | `[1, 16]` | `[1, 16]` | 73.28 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_12 | 168 | 6648 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 15.36 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_120 | 1680 | 69720 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 139.77 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_180 | 2520 | 104760 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 277.51 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_25 | 350 | 14240 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 30.24 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_250 | 3500 | 145640 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 295.95 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_3 | 42 | 1392 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 4.4 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_50 | 700 | 28840 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 56.78 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_6 | 84 | 3144 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 7.74 ms |
| grade_1_poorfit | `sub_sram_tiny_cnn` | scale_80 | 1120 | 46360 | `[1, 3, 32, 32]` | `[1, 8, 32, 32]` | 91.0 ms |
| grade_1_poorfit | `tiny_bn_relu_chain` | scale_10 | 250 | 3040 | `[1, 16, 32, 32]` | `[1, 16, 32, 32]` | 20.55 ms |
| grade_1_poorfit | `tiny_bn_relu_chain` | scale_140 | 3500 | 42560 | `[1, 16, 32, 32]` | `[1, 16, 32, 32]` | 338.09 ms |
| grade_1_poorfit | `tiny_bn_relu_chain` | scale_20 | 500 | 6080 | `[1, 16, 32, 32]` | `[1, 16, 32, 32]` | 47.65 ms |
| grade_1_poorfit | `tiny_bn_relu_chain` | scale_200 | 5000 | 60800 | `[1, 16, 32, 32]` | `[1, 16, 32, 32]` | 470.89 ms |
| grade_1_poorfit | `tiny_bn_relu_chain` | scale_4 | 100 | 1216 | `[1, 16, 32, 32]` | `[1, 16, 32, 32]` | 10.91 ms |
| grade_1_poorfit | `tiny_bn_relu_chain` | scale_45 | 1125 | 13680 | `[1, 16, 32, 32]` | `[1, 16, 32, 32]` | 112.35 ms |
| grade_1_poorfit | `tiny_bn_relu_chain` | scale_90 | 2250 | 27360 | `[1, 16, 32, 32]` | `[1, 16, 32, 32]` | 171.98 ms |
| grade_2_lowfit | `deep_dwconv` | scale_12 | 324 | 213504 | `[1, 128, 32, 32]` | `[1, 128, 32, 32]` | 30.8 ms |
| grade_2_lowfit | `deep_dwconv` | scale_120 | 3240 | 2135040 | `[1, 128, 32, 32]` | `[1, 128, 32, 32]` | 377.62 ms |
| grade_2_lowfit | `deep_dwconv` | scale_180 | 4860 | 3202560 | `[1, 128, 32, 32]` | `[1, 128, 32, 32]` | 623.78 ms |
| grade_2_lowfit | `deep_dwconv` | scale_25 | 675 | 444800 | `[1, 128, 32, 32]` | `[1, 128, 32, 32]` | 80.05 ms |
| grade_2_lowfit | `deep_dwconv` | scale_5 | 135 | 88960 | `[1, 128, 32, 32]` | `[1, 128, 32, 32]` | 18.82 ms |
| grade_2_lowfit | `deep_dwconv` | scale_50 | 1350 | 889600 | `[1, 128, 32, 32]` | `[1, 128, 32, 32]` | 287.05 ms |
| grade_2_lowfit | `deep_dwconv` | scale_80 | 2160 | 1423360 | `[1, 128, 32, 32]` | `[1, 128, 32, 32]` | 234.22 ms |
| grade_2_lowfit | `fc_classifier_chain` | scale_12 | 75 | 3151872 | `[1, 512]` | `[1, 512]` | 11.15 ms |
| grade_2_lowfit | `fc_classifier_chain` | scale_20 | 123 | 5253120 | `[1, 512]` | `[1, 512]` | 17.93 ms |
| grade_2_lowfit | `fc_classifier_chain` | scale_3 | 21 | 787968 | `[1, 512]` | `[1, 512]` | 3.35 ms |
| grade_2_lowfit | `fc_classifier_chain` | scale_35 | 213 | 9192960 | `[1, 512]` | `[1, 512]` | 22.56 ms |
| grade_2_lowfit | `fc_classifier_chain` | scale_6 | 39 | 1575936 | `[1, 512]` | `[1, 512]` | 4.23 ms |
| grade_2_lowfit | `fc_classifier_chain` | scale_60 | 363 | 15759360 | `[1, 512]` | `[1, 512]` | 40.41 ms |
| grade_2_lowfit | `fc_classifier_chain` | scale_90 | 543 | 23639040 | `[1, 512]` | `[1, 512]` | 60.99 ms |
| grade_2_lowfit | `lnorm_transformer_chain` | scale_110 | 1870 | 7293440 | `[1, 256]` | `[1, 256]` | 157.2 ms |
| grade_2_lowfit | `lnorm_transformer_chain` | scale_160 | 2720 | 10608640 | `[1, 256]` | `[1, 256]` | 410.44 ms |
| grade_2_lowfit | `lnorm_transformer_chain` | scale_18 | 306 | 1193472 | `[1, 256]` | `[1, 256]` | 24.34 ms |
| grade_2_lowfit | `lnorm_transformer_chain` | scale_3 | 51 | 198912 | `[1, 256]` | `[1, 256]` | 4.15 ms |
| grade_2_lowfit | `lnorm_transformer_chain` | scale_40 | 680 | 2652160 | `[1, 256]` | `[1, 256]` | 52.08 ms |
| grade_2_lowfit | `lnorm_transformer_chain` | scale_70 | 1190 | 4641280 | `[1, 256]` | `[1, 256]` | 90.49 ms |
| grade_2_lowfit | `lnorm_transformer_chain` | scale_8 | 136 | 530432 | `[1, 256]` | `[1, 256]` | 10.13 ms |
| grade_2_lowfit | `sequential_vgg` | scale_10 | 105 | 334144 | `[1, 3, 64, 64]` | `[1, 64, 64, 64]` | 29.37 ms |
| grade_2_lowfit | `sequential_vgg` | scale_130 | 1305 | 4765504 | `[1, 3, 64, 64]` | `[1, 64, 64, 64]` | 337.18 ms |
| grade_2_lowfit | `sequential_vgg` | scale_200 | 2005 | 7350464 | `[1, 3, 64, 64]` | `[1, 64, 64, 64]` | 570.23 ms |
| grade_2_lowfit | `sequential_vgg` | scale_22 | 225 | 777280 | `[1, 3, 64, 64]` | `[1, 64, 64, 64]` | 51.79 ms |
| grade_2_lowfit | `sequential_vgg` | scale_300 | 3005 | 11043264 | `[1, 3, 64, 64]` | `[1, 64, 64, 64]` | 792.99 ms |
| grade_2_lowfit | `sequential_vgg` | scale_4 | 45 | 112576 | `[1, 3, 64, 64]` | `[1, 64, 64, 64]` | 10.09 ms |
| grade_2_lowfit | `sequential_vgg` | scale_45 | 455 | 1626624 | `[1, 3, 64, 64]` | `[1, 64, 64, 64]` | 106.44 ms |
| grade_2_lowfit | `sequential_vgg` | scale_80 | 805 | 2919104 | `[1, 3, 64, 64]` | `[1, 64, 64, 64]` | 236.24 ms |
| grade_3_moderate | `bottleneck_resnet` | scale_110 | 5170 | 503360 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 617.09 ms |
| grade_3_moderate | `bottleneck_resnet` | scale_12 | 564 | 54912 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 53.99 ms |
| grade_3_moderate | `bottleneck_resnet` | scale_2 | 94 | 9152 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 10.22 ms |
| grade_3_moderate | `bottleneck_resnet` | scale_25 | 1175 | 114400 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 124.02 ms |
| grade_3_moderate | `bottleneck_resnet` | scale_45 | 2115 | 205920 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 220.77 ms |
| grade_3_moderate | `bottleneck_resnet` | scale_5 | 235 | 22880 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 22.79 ms |
| grade_3_moderate | `bottleneck_resnet` | scale_70 | 3290 | 320320 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 340.89 ms |
| grade_3_moderate | `resnet_basic` | scale_100 | 5600 | 7411200 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 798.32 ms |
| grade_3_moderate | `resnet_basic` | scale_12 | 672 | 889344 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 93.97 ms |
| grade_3_moderate | `resnet_basic` | scale_150 | 8400 | 11116800 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 1198.85 ms |
| grade_3_moderate | `resnet_basic` | scale_2 | 112 | 148224 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 16.04 ms |
| grade_3_moderate | `resnet_basic` | scale_25 | 1400 | 1852800 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 198.32 ms |
| grade_3_moderate | `resnet_basic` | scale_45 | 2520 | 3335040 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 376.08 ms |
| grade_3_moderate | `resnet_basic` | scale_5 | 280 | 370560 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 36.12 ms |
| grade_3_moderate | `resnet_basic` | scale_70 | 3920 | 5187840 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 515.07 ms |
| grade_3_moderate | `two_branch_add` | scale_130 | 3900 | 18120960 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 939.0 ms |
| grade_3_moderate | `two_branch_add` | scale_18 | 540 | 2509056 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 137.84 ms |
| grade_3_moderate | `two_branch_add` | scale_3 | 90 | 418176 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 23.52 ms |
| grade_3_moderate | `two_branch_add` | scale_35 | 1050 | 4878720 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 248.71 ms |
| grade_3_moderate | `two_branch_add` | scale_60 | 1800 | 8363520 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 459.01 ms |
| grade_3_moderate | `two_branch_add` | scale_8 | 240 | 1115136 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 54.22 ms |
| grade_3_moderate | `two_branch_add` | scale_90 | 2700 | 12545280 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | 674.31 ms |
| grade_4_highfit | `inception_multibranch` | scale_10 | 860 | 365280 | `[1, 128, 64, 64]` | `[1, 128, 64, 64]` | 149.94 ms |
| grade_4_highfit | `inception_multibranch` | scale_2 | 172 | 73056 | `[1, 128, 64, 64]` | `[1, 128, 64, 64]` | 32.46 ms |
| grade_4_highfit | `inception_multibranch` | scale_20 | 1720 | 730560 | `[1, 128, 64, 64]` | `[1, 128, 64, 64]` | 326.48 ms |
| grade_4_highfit | `inception_multibranch` | scale_35 | 3010 | 1278480 | `[1, 128, 64, 64]` | `[1, 128, 64, 64]` | 586.22 ms |
| grade_4_highfit | `inception_multibranch` | scale_5 | 430 | 182640 | `[1, 128, 64, 64]` | `[1, 128, 64, 64]` | 84.24 ms |
| grade_4_highfit | `inception_multibranch` | scale_55 | 4730 | 2009040 | `[1, 128, 64, 64]` | `[1, 128, 64, 64]` | 891.14 ms |
| grade_4_highfit | `inception_multibranch` | scale_80 | 6880 | 2922240 | `[1, 128, 64, 64]` | `[1, 128, 64, 64]` | 1350.61 ms |
| grade_5_bestfit | `bev_perception_512` | scale_15 | 210 | 2140032 | `[1, 64, 128, 128]` | `[1, 128, 128, 128]` | 320.73 ms |
| grade_5_bestfit | `bev_perception_512` | scale_30 | 420 | 4353792 | `[1, 64, 128, 128]` | `[1, 128, 128, 128]` | 785.88 ms |
| grade_5_bestfit | `bev_perception_512` | scale_4 | 56 | 516608 | `[1, 64, 128, 128]` | `[1, 128, 128, 128]` | 94.33 ms |
| grade_5_bestfit | `bev_perception_512` | scale_50 | 700 | 7305472 | `[1, 64, 128, 128]` | `[1, 128, 128, 128]` | 1203.52 ms |
| grade_5_bestfit | `bev_perception_512` | scale_8 | 112 | 1106944 | `[1, 64, 128, 128]` | `[1, 128, 128, 128]` | 193.03 ms |

## Các mô hình bị bỏ qua (Cần export TorchScript)

| Grade | Kiến trúc | Scale | Lý do |
| :--- | :--- | :--- | :--- |
| grade_4_highfit | `fpn_simple` | scale_2 | SKIPPED_NO_TRACED |
| grade_4_highfit | `fpn_simple` | scale_3 | SKIPPED_NO_TRACED |
| grade_4_highfit | `fpn_simple` | scale_4 | SKIPPED_NO_TRACED |
| grade_4_highfit | `fpn_simple` | scale_5 | SKIPPED_NO_TRACED |
| grade_4_highfit | `fpn_simple` | scale_6 | SKIPPED_NO_TRACED |
| grade_4_highfit | `unet_skip` | depth_2 | SKIPPED_NO_TRACED |
| grade_4_highfit | `unet_skip` | depth_3 | SKIPPED_NO_TRACED |
| grade_4_highfit | `unet_skip` | depth_4 | SKIPPED_NO_TRACED |
| grade_4_highfit | `unet_skip` | depth_5 | SKIPPED_NO_TRACED |
| grade_5_bestfit | `bev_perception_512` | scale_80 | SKIPPED_NO_TRACED |
| grade_5_bestfit | `yolo_fpn_panet` | scale_10 | SKIPPED_NO_TRACED |
| grade_5_bestfit | `yolo_fpn_panet` | scale_2 | SKIPPED_NO_TRACED |
| grade_5_bestfit | `yolo_fpn_panet` | scale_20 | SKIPPED_NO_TRACED |
| grade_5_bestfit | `yolo_fpn_panet` | scale_35 | SKIPPED_NO_TRACED |
| grade_5_bestfit | `yolo_fpn_panet` | scale_5 | SKIPPED_NO_TRACED |
| grade_5_bestfit | `yolo_fpn_panet` | scale_55 | SKIPPED_NO_TRACED |
| grade_5_bestfit | `yolo_fpn_panet` | scale_80 | SKIPPED_NO_TRACED |
