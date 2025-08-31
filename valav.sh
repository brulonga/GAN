set +e  

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan0.yml --name srnet+GAN_01 || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan1.yml --name srnet+LPIPS_05 || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan2.yml --name srnet+GAN_005 || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan3.yml --name srnet+LPIPS_1_GAN_005 || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan4.yml --name srnet || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan5.yml --name srnet+GAN_01_no_rel || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan5.yml --name srnet+GAN_01_no_rel_pretrained || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan0.yml --name srnet+GAN_01_float32 || true
