#!/usr/bin/env bash
# Recorded per-job commands; chronological interleaving, not a concurrency replay.
# For the original parallel run use scripts/run_replay.py as README describes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# 2026-10-26T06:45:15Z count_1_103_-1.87 rc=0 wall_s=4.4 stdout_sha256=3765f1f4e728c4b688ef85b1be24d4642aebe19faaa0eaf4016ee14eb3e769c3
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/count_check.py 1 103 -1.87)
# 2026-10-26T06:45:15Z rs_validate rc=0 wall_s=15.6 stdout_sha256=a277d6463b6b3b8b3b2a91a8cd9b9118fa287ee6758c2e8ebafe5bb30c3e8c34
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/rs_zeta.py validate)
# 2026-10-26T06:45:15Z test_numeric rc=0 wall_s=2.2 stdout_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B -m unittest -v test_exact test_numeric)
# 2026-10-26T06:45:15Z validate_H0 rc=0 wall_s=6.1 stdout_sha256=ddb1869295287b77d971b66c8f369ddf836e70863ffda4603f7e001d854f36a2
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/validate_H0.py)
# 2026-10-26T06:45:17Z count_1_103_-1.89 rc=0 wall_s=4.5 stdout_sha256=4a3542afc19edbde9fb308e3f76dc227e0e37a9aacacb97b5a77e82e2716494d
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/count_check.py 1 103 -1.89)
# 2026-10-26T06:45:20Z count_1_103_0 rc=0 wall_s=4.4 stdout_sha256=eee87ce75a838aa6fff0b59e47243f40567a4e25bee205e48fcb85189ca2022d
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/count_check.py 1 103 0)
# 2026-10-26T06:45:21Z count_1_270_-0.38 rc=0 wall_s=11.0 stdout_sha256=d0f6a2f3e8779c42fe2072ce5f07fa54207c2b2617388cabd21f872ec8114aa2
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/count_check.py 1 270 -0.38)
# 2026-10-26T06:45:22Z count_1_270_-0.39 rc=0 wall_s=11.1 stdout_sha256=3ec1f175d82cedee5510cf528d29edc5a4c3209df21b7aeda9ca728295e033a0
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/count_check.py 1 270 -0.39)
# 2026-10-26T06:45:24Z count_1_270_0 rc=0 wall_s=10.6 stdout_sha256=abc141b59fd80f9153099064edaac2f1df2d457560fe8109c94d1077e5789b0d
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/count_check.py 1 270 0)
# 2026-10-26T06:45:31Z rs_A rc=0 wall_s=4.5 stdout_sha256=b395e10099a1379d788819daeea13b7eae2baca28c0039a7591ce0975e6f142e
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/rs_zeta.py A 1e4 1e5)
# 2026-10-26T06:45:32Z check_dbn_basic rc=0 wall_s=49.6 stdout_sha256=9d14d5d8d0f3e0fd29124adfb0276f366deb3597500e078c8c878131c4c1fc85
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/check_dbn_basic.py)
# 2026-10-26T06:45:33Z check_collision rc=0 wall_s=77.6 stdout_sha256=c557a8c2570533a9a9911dbbe48ad8c332995eda3b10986b8cf0c83c368a6919
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/check_collision.py)
# 2026-10-26T06:45:35Z refine_check rc=0 wall_s=21.0 stdout_sha256=8439bc7125a3b0a8fcc75dd769088eaa94ce96ea9defb9db10329f6e47fdc3e3
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/refine_check.py)
# 2026-10-26T06:45:35Z rs_B rc=0 wall_s=99.1 stdout_sha256=ac04318c1b9d9b92ffb0b7910e370e9ff77946bac04092ed0de9c1d876677695
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/rs_zeta.py B 1e5 1e6)
# 2026-10-26T06:45:56Z indep_collision rc=0 wall_s=2.4 stdout_sha256=b5505605e28868b53fc09d28aa8e29a88d098ed1d5d38d675cec9b93b2d91a56
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/indep_collision.py)
# 2026-10-26T06:45:58Z kappa_check rc=0 wall_s=0.5 stdout_sha256=36218514bd6534d1c6d06cc1c71f4d42366dc1312151e9de2a536ac2323dbb95
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/kappa_check.py)
# 2026-10-26T06:45:59Z kappa_indep_degree6 rc=0 wall_s=0.5 stdout_sha256=c9ee772c7767780d8834f70e5cc79769c6af9aad1cb7adfd921eac28110ff26d
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/kappa_indep_degree6.py)
# 2026-10-26T06:45:59Z kappa_indep_random rc=0 wall_s=0.8 stdout_sha256=161d26be0f78d53159865b6f29d18ddc508db16d9e21daa60854028d98c5d2ae
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/kappa_indep_random.py)
# 2026-10-26T06:46:00Z zero_velocity rc=0 wall_s=4.7 stdout_sha256=b3ae9e692cd049260b9ef91c5dc4e855eb012430ea15bf195766db3037075ac3
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/zero_velocity.py)
# 2026-10-26T06:46:05Z indep_ht_xi rc=0 wall_s=7.5 stdout_sha256=e8e373471f000d5e10c165592b23dcff8fa447712a80fe410e56dd46bbceadc4
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/indep_ht.py 60 xi)
# 2026-10-26T06:46:05Z smoke_hflow rc=0 wall_s=0.8 stdout_sha256=510a8c801ccbcd0a223578ccf47c06d6c3acfe00c1fdf5e23e56152c7c362d0e
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/smoke_hflow.py)
# 2026-10-26T06:46:13Z indep_ht_c910 rc=0 wall_s=83.7 stdout_sha256=71da411fe35b93375dd1d1e27836e74b67a88aaa77b1ae9ccae71386fe94f00c
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/indep_ht.py 60 c910)
# 2026-10-26T06:46:22Z indep_ht_c3435 rc=0 wall_s=365.0 stdout_sha256=bdfeb91fe96a91d6d12f5534c839e9a3c611b95f85abf67167ec8167bb102574
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/indep_ht.py 60 c3435)
# 2026-10-26T06:46:51Z lobe_identity rc=0 wall_s=286.3 stdout_sha256=82510fd5378466aa32c8be2c463fb20f0ecc702bcec2257437481e2c7bd38cc4
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/lobe_identity.py)
# 2026-10-26T06:47:14Z gauss rc=0 wall_s=10.2 stdout_sha256=62338f7279c3ff6ca9c37fe9a451f2a779445032bf843b58adcffc85aeebc1c2
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/gauss_proc.py 500 4)
# 2026-10-26T06:47:25Z cue12 rc=0 wall_s=4.1 stdout_sha256=ffe479f26064b8c9b7f38d6f6cd3b3c800fb4dfb6afa225b24db67b36d351f62
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/cue.py 12 86016 4)
# 2026-10-26T06:47:29Z cue128 rc=0 wall_s=41.1 stdout_sha256=fc64e6fd9369068df2217a3fc24d694ef785e0705a13dd89fc1c433922cedaf6
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/cue.py 128 8064 4)
# 2026-10-26T06:47:37Z track_z270 rc=0 wall_s=150.1 stdout_sha256=adced02e0e476834960db80ee45f962474a4f4514dfb739d669b86911c4aa99a
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/track.py 270)
# 2026-10-26T06:48:10Z analyze rc=0 wall_s=27.5 stdout_sha256=32d1384826d72e29032c277906bf11424a63c008c57cf10366c634c00ae310cb
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/analyze.py)
# 2026-10-26T06:48:37Z matched_tail rc=0 wall_s=4.7 stdout_sha256=afda9943d6d4a122fe88f4cc23ba1191c7e3a3ce77f38d86a584665c3b164302
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/matched_tail.py)
# 2026-10-26T06:48:42Z make_plots rc=0 wall_s=1.1 stdout_sha256=dc51b8c96c2d745df3bd5590d990230a482fd247123599548e0632fdbf97fc22
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/make_plots.py)
# 2026-10-26T06:48:43Z sample_600 rc=0 wall_s=124.5 stdout_sha256=97b55e9d8c8a47874e099a88de8d491bc69afb386716c4c36ec89c33745a8d88
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/sample_Z.py 600 900 0.02)
# 2026-10-26T06:50:07Z track_low10 rc=0 wall_s=49.8 stdout_sha256=76f4f56b14f35182d5d0b3dcc742801d1e507d4077c1257cca51ecd397f67b8d
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/track.py 103 -40 0.02 low10)
# 2026-10-26T06:50:48Z sample_5000 rc=0 wall_s=217.5 stdout_sha256=2aaef5dff971cca635f6cdc7e066acdc6cd9a64a0608ec60c3db409808ba3267
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/sample_Z.py 5000 5200 0.015)
# 2026-10-26T06:50:56Z track_dt0.01 rc=0 wall_s=33.7 stdout_sha256=418c0914407f5d64ae493b1a6f1bf77de9f51dde1323c63c8988a5366e3f893b
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/track.py 103 -40 0.01 dt0.01)
# 2026-10-26T06:51:30Z track_dt0.02 rc=0 wall_s=52.8 stdout_sha256=76f4f56b14f35182d5d0b3dcc742801d1e507d4077c1257cca51ecd397f67b8d
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/track.py 103 -40 0.02 dt0.02)
# 2026-10-26T06:51:37Z track_dt0.025 rc=0 wall_s=52.4 stdout_sha256=61dbd3082df5742800db08abd8744a7f876d4e2c83e8165debd626f11911353d
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/track.py 103 -40 0.025 dt0.025)
# 2026-10-26T06:52:23Z track_dt0.04 rc=0 wall_s=17.2 stdout_sha256=f0295474cd4d9a15195834274f4010633556c1faab8a33c0328b44901b56792c
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/track.py 103 -40 0.04 dt0.04)
# 2026-10-26T06:52:27Z track_dt0.05 rc=0 wall_s=30.2 stdout_sha256=396cb95bd568f0b12704f2a2f72291ff9bdec15b3574450f3786eb0d29142470
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/track.py 103 -40 0.05 dt0.05)
# 2026-10-26T06:52:29Z check_theta rc=0 wall_s=1.5 stdout_sha256=99cc3b26b055c8540f96daf1864e93c270269789b4d5fdc8430eeadddeef49b1
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B theta/check_theta.py)
# 2026-10-26T06:52:31Z project_theta_checks rc=0 wall_s=236.0 stdout_sha256=9e848d419dc33fdc83ab1a42c384861519634694a7b510a77de20c51224f2253
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B theta/project_theta_checks.py)
# 2026-10-26T06:52:40Z check_gue_transfer rc=0 wall_s=1.1 stdout_sha256=52ba80c24813d0de577e4e495202ac0aa2d314e7c4a7cc1c87ed89611b603160
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/check_gue_transfer.py)
# 2026-10-26T06:52:41Z check_collision_persistence rc=0 wall_s=37.0 stdout_sha256=a64b3d1aaf7d305bb55486eae86ea06255441716da810b697a00a85998b9115f
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/check_collision_persistence.py)
# 2026-10-26T06:52:41Z fold_controls rc=0 wall_s=0.1 stdout_sha256=8abb61b16d4961791c80a04c7e98971581cc6dc9e73af5d66a2f85a91dc8e381
(cd /workspace/scratch/57388d5ddbb4/zeta-run && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B experiments/universality/fold_controls.py)
# 2026-10-26T06:52:57Z collision_table rc=0 wall_s=0.1 stdout_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/collision_table.py)
# 2026-10-26T06:52:58Z fold_pair34_35 rc=0 wall_s=10.0 stdout_sha256=524ae4454bcb454360a9be4467b2acdf133e02a5664198fae4eb370d4cdbf143
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/fold_check.py -0.3845611166 223.2757801856 pair34_35)
# 2026-10-26T06:53:08Z fold_pair9_10 rc=0 wall_s=3.2 stdout_sha256=52118e6ad0150f876f6699ec7a7ee8a42c1f6e35ee750f31ede423e67daffc91
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/fold_check.py -1.8779090730 99.3284693056 pair9_10)
# 2026-10-26T06:53:12Z fold_analyze rc=0 wall_s=0.4 stdout_sha256=db062af9bdd2fa4374c5893b2d99216706d7a3959a38434b24709521824185a1
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B dbn/fold_analyze.py pair34_35 pair9_10)
# 2026-10-26T06:54:25Z checkZ_600 rc=0 wall_s=0.6 stdout_sha256=2fe69b3e343c6d59fa1436323bef3caadc6f3a2dd45951b7d52cfa2142fc9f08
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/check_Z_persistence.py Z_grid_600_900.npz)
# 2026-10-26T06:54:26Z barcode_bruteforce rc=0 wall_s=0.6 stdout_sha256=872e4cf0d0e159898555a2fddd5f187d810f6305492fa81f62426627e4bdafbe
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/barcode_bruteforce.py)
# 2026-10-26T06:54:26Z checkZ_5000 rc=0 wall_s=0.7 stdout_sha256=cc939a7526087e28207ac7f81e74457d713fdf3ae238ffbbdac837cff75832d8
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/check_Z_persistence.py Z_grid_5000_5200.npz)
# 2026-10-26T06:54:26Z check_gauss rc=0 wall_s=1.3 stdout_sha256=00459801ce5a808a385a9cc5a10407d9512a9189189f95fe31991d941702dbc6
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/check_gauss.py)
# 2026-10-26T06:54:26Z check_zeta_quad rc=0 wall_s=15.7 stdout_sha256=4264fb16d2e93eec08c9f46858f9e9d1f9b89f1cfcd83b88d7240e3c86cd3ba6
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/check_zeta_quad.py)
# 2026-10-26T06:54:27Z gauss_hidden rc=0 wall_s=15.2 stdout_sha256=cd01926917288241fcbf3c70d42131ceb0861915a30ecaa97986f939e69b89f0
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/gauss_hidden.py)
# 2026-10-26T06:54:28Z localamp_baseline rc=0 wall_s=6.4 stdout_sha256=41653c143ac3a2f84d3dfa45ad7073ac14dbfcb2a87f9dcbbcc2a50d9957c3eb
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/localamp_baseline.py)
# 2026-10-26T06:54:34Z cue_gap_shape rc=0 wall_s=15.1 stdout_sha256=536f1caed61287f07a329eea8337c0134b9f4bec1afb8c331a1e6ee2761f7f7f
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/cue_gap_shape.py)
# 2026-10-26T06:54:42Z cue_endpoint_check rc=0 wall_s=0.6 stdout_sha256=12e3e8b948b11b8eb8f1fc2d72c2478a17c978defa60cd15b8a230e88ed9aa35
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/cue_endpoint_check.py)
# 2026-10-26T06:54:42Z rs_spot_check rc=0 wall_s=1.6 stdout_sha256=c16950bf458312547ac05e92f993b74ab1a30d4b779bf4eaab0ab7405dcd550f
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/rs_spot_check.py)
# 2026-10-26T06:54:43Z block_sensitivity rc=0 wall_s=3.9 stdout_sha256=d56f54dd860e43d5153252aafdd2d61e03690885fbc399ab5f21cdf35791d5cd
(cd /workspace/scratch/57388d5ddbb4/zeta-run/experiments/zeta_fold_exploration && /workspace/scratch/57388d5ddbb4/zeta-venv/bin/python -B stats/block_sensitivity.py)
