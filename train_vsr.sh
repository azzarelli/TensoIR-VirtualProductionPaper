export PYTHONPATH=. && python train_tensoIR_general_multi_lights.py  --config ./configs/multi_light_general/vsr.txt --relight_chunk_size 10000

# export PYTHONPATH=. && python train_tensoIR_general_multi_lights.py --config ./configs/multi_light_general/vsr.txt --ckpt "./log/log_vsr_multilight/VSR_3.1_1-20260329-201358/checkpoints/VSR_3.1_1_70000.th" --render_only 1 --render_test 1