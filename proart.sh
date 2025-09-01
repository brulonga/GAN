set +e  

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan_stg1.yml --name esrgan_stg1 || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan_stg2.yml --name esrgan_stg2 || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan_stg3.yml --name esrgan_stg3 || true