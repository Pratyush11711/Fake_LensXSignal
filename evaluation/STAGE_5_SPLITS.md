# STAGE 5 — BENCHMARK SPLITS SPECIFICATION

> [!IMPORTANT]
> **DISJOINT SPLIT INTEGRITY GUARANTEE:**
> The second benchmark dataset (DeepFake Detection Challenge, DFDC) is partitioned into two strictly disjoint subsets: **Calibration / Validation ($N=50$)** and **Final Test ($N=50$)**.
> Zero video overlap exists between Calibration and Final Test.
> The 100 Google/Jigsaw DFD benchmark videos from Stage 4B remain completely outside both new splits.

---

## 1. Dataset Overview
- **Benchmark Name**: DeepFake Detection Challenge (DFDC) Balanced Subset
- **Source Repository**: `gonnerthetooner/DFDC-extracted-full` (`dfdc_train_part_00/dfdc_train_part_0/`)
- **Official Metadata**: Extracted directly from official `metadata.json`
- **Total Videos**: 100 (50 REAL, 50 FAKE)
- **Storage Location**: `D:/benchmark_dataset_dfdc/`

---

## 2. Calibration / Validation Subset ($N = 50$ Videos)
Used strictly for fitting temperature scaling parameter $T$ and exploratory threshold characterization.
Never used for final reporting.

| ID | Filename | Ground Truth | Original Source Pairing |
| :--- | :--- | :---: | :--- |
| `dfdc_real_001` | `vpmyeepbep.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_002` | `fzvpbrzssi.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_003` | `syxobtuucp.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_004` | `dhjnjkzuhq.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_005` | `xcruhaccxc.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_006` | `vtunvalyji.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_007` | `qyqufaskjs.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_008` | `rnpefxwptv.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_009` | `sttnfyptum.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_010` | `xugmhbetrw.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_011` | `uqtqhiqymz.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_012` | `jawgcggquk.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_013` | `yexeazbqig.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_014` | `hivnldfvyl.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_015` | `wfzjxzhdkj.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_016` | `apedduehoy.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_017` | `exseruhiuk.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_018` | `lnhkjhyhvw.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_019` | `prdrkaxeob.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_020` | `pqdeutauqc.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_021` | `xjzkfqddyk.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_022` | `fufcmupzen.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_023` | `lokzwdldxp.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_024` | `hplxtssgnz.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_025` | `gcdtglsoqj.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_fake_001` | `owxbbpjpch.mp4` | **FAKE** | `wynotylpnm.mp4` |
| `dfdc_fake_002` | `htorvhbcae.mp4` | **FAKE** | `wclvkepakb.mp4` |
| `dfdc_fake_003` | `fckxaqjbxk.mp4` | **FAKE** | `vpmyeepbep.mp4` |
| `dfdc_fake_004` | `sphirandia.mp4` | **FAKE** | `jawgcggquk.mp4` |
| `dfdc_fake_005` | `vsmadeuczx.mp4` | **FAKE** | `qarqtkvgby.mp4` |
| `dfdc_fake_006` | `ohaqlzfnuv.mp4` | **FAKE** | `sttnfyptum.mp4` |
| `dfdc_fake_007` | `komngcqveq.mp4` | **FAKE** | `kqlvggiqee.mp4` |
| `dfdc_fake_008` | `iafvzgpbix.mp4` | **FAKE** | `lzbmwwejxb.mp4` |
| `dfdc_fake_009` | `uaukglhmje.mp4` | **FAKE** | `jytrvwlewz.mp4` |
| `dfdc_fake_010` | `dtjcyzgdts.mp4` | **FAKE** | `wcqvzujamg.mp4` |
| `dfdc_fake_011` | `viuioldtnu.mp4` | **FAKE** | `exxqlfpnbz.mp4` |
| `dfdc_fake_012` | `wnaweyzlqh.mp4` | **FAKE** | `vpmyeepbep.mp4` |
| `dfdc_fake_013` | `ejhhdlzpjg.mp4` | **FAKE** | `nweufafotd.mp4` |
| `dfdc_fake_014` | `sxyrkshzsg.mp4` | **FAKE** | `jvtjxreizj.mp4` |
| `dfdc_fake_015` | `uvfkppqsjy.mp4` | **FAKE** | `gvasarkpfh.mp4` |
| `dfdc_fake_016` | `kedahvzait.mp4` | **FAKE** | `wclvkepakb.mp4` |
| `dfdc_fake_017` | `idczhqbbqz.mp4` | **FAKE** | `prmwoaeeng.mp4` |
| `dfdc_fake_018` | `clzkcmoyhb.mp4` | **FAKE** | `gnmmhlbzge.mp4` |
| `dfdc_fake_019` | `vpjfmetsvn.mp4` | **FAKE** | `nweufafotd.mp4` |
| `dfdc_fake_020` | `muksyyltmg.mp4` | **FAKE** | `fopjiyxiqd.mp4` |
| `dfdc_fake_021` | `ztwlbdwyni.mp4` | **FAKE** | `kqlvggiqee.mp4` |
| `dfdc_fake_022` | `jaurxwocly.mp4` | **FAKE** | `pqdeutauqc.mp4` |
| `dfdc_fake_023` | `jkxjvuioek.mp4` | **FAKE** | `wfzjxzhdkj.mp4` |
| `dfdc_fake_024` | `rfpjsloxzg.mp4` | **FAKE** | `dxfdovivlw.mp4` |
| `dfdc_fake_025` | `kmcdjxmnoa.mp4` | **FAKE** | `sttnfyptum.mp4` |

---

## 3. Final Test Subset ($N = 50$ Videos)
Held out completely until calibration parameter $T$ is frozen.
Used for final unbiased comparative evaluation of Baseline ViT and GenD PE_L (raw vs calibrated).

| ID | Filename | Ground Truth | Original Source Pairing |
| :--- | :--- | :---: | :--- |
| `dfdc_real_026` | `wclvkepakb.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_027` | `upmgtackuf.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_028` | `rmuxlgsedw.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_029` | `lnjkpdviqb.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_030` | `hicjuubiau.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_031` | `fsaronfupy.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_032` | `prmwoaeeng.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_033` | `ucthmsajay.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_034` | `fopjiyxiqd.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_035` | `jytrvwlewz.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_036` | `rmufsuogzn.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_037` | `nrnklcxdzq.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_038` | `dxfdovivlw.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_039` | `srfefmyjvt.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_040` | `eiwtggvtfp.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_041` | `fgobmbcami.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_042` | `hsbwhlolsn.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_043` | `ldtgofdaqg.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_044` | `rktrpsdlci.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_045` | `heiyoojifp.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_046` | `uprwuohbwx.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_047` | `wynotylpnm.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_048` | `yxvmusxvcz.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_049` | `xmkwsnuzyq.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_real_050` | `fdpisghkmd.mp4` | **REAL** | `N/A (Real source)` |
| `dfdc_fake_026` | `vvwuqslvnq.mp4` | **FAKE** | `ddtbarpcgo.mp4` |
| `dfdc_fake_027` | `rpxnyljmsf.mp4` | **FAKE** | `nweufafotd.mp4` |
| `dfdc_fake_028` | `txxungvxms.mp4` | **FAKE** | `lzbmwwejxb.mp4` |
| `dfdc_fake_029` | `ihnhxathkq.mp4` | **FAKE** | `sttnfyptum.mp4` |
| `dfdc_fake_030` | `lobxktzhbg.mp4` | **FAKE** | `dhjnjkzuhq.mp4` |
| `dfdc_fake_031` | `slhinthqcq.mp4` | **FAKE** | `wfzjxzhdkj.mp4` |
| `dfdc_fake_032` | `hqzwudvhih.mp4` | **FAKE** | `xmkwsnuzyq.mp4` |
| `dfdc_fake_033` | `xpzfhhwkwb.mp4` | **FAKE** | `xmkwsnuzyq.mp4` |
| `dfdc_fake_034` | `sglzgdcjkb.mp4` | **FAKE** | `vokrpfjpeb.mp4` |
| `dfdc_fake_035` | `xxvwbfmgzy.mp4` | **FAKE** | `doniqevxeg.mp4` |
| `dfdc_fake_036` | `dtmlspmqqp.mp4` | **FAKE** | `aayrffkzxn.mp4` |
| `dfdc_fake_037` | `hsxrmrcaqh.mp4` | **FAKE** | `upmgtackuf.mp4` |
| `dfdc_fake_038` | `ykbhmueovi.mp4` | **FAKE** | `prmwoaeeng.mp4` |
| `dfdc_fake_039` | `inzrlbgtul.mp4` | **FAKE** | `nrnklcxdzq.mp4` |
| `dfdc_fake_040` | `xtcbnasaew.mp4` | **FAKE** | `jzupayeuln.mp4` |
| `dfdc_fake_041` | `kgfcupxgur.mp4` | **FAKE** | `ddtbarpcgo.mp4` |
| `dfdc_fake_042` | `nkojmntmdg.mp4` | **FAKE** | `exxqlfpnbz.mp4` |
| `dfdc_fake_043` | `btizoxwqwp.mp4` | **FAKE** | `fnslimfagb.mp4` |
| `dfdc_fake_044` | `zvsivkdkak.mp4` | **FAKE** | `dhjnjkzuhq.mp4` |
| `dfdc_fake_045` | `yjllqhbmke.mp4` | **FAKE** | `nrnklcxdzq.mp4` |
| `dfdc_fake_046` | `ygknkubeyb.mp4` | **FAKE** | `nweufafotd.mp4` |
| `dfdc_fake_047` | `wkczijuamz.mp4` | **FAKE** | `jytrvwlewz.mp4` |
| `dfdc_fake_048` | `dolynaukae.mp4` | **FAKE** | `gnmmhlbzge.mp4` |
| `dfdc_fake_049` | `eisofhptvk.mp4` | **FAKE** | `nweufafotd.mp4` |
| `dfdc_fake_050` | `xymeodlrhb.mp4` | **FAKE** | `jzupayeuln.mp4` |
