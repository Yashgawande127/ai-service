# Virtual Try-On AI Service: GPU Inference & Evaluation Report

## 1. Overview & Setup

This repository contains the AI Virtual Try-On service focused on Indian Ethnic Wear (Sarees, Kurtis, Lehengas) and Western Wear, evaluating both **Mask-Free CatVTON** (`CatVTONPix2PixPipeline`) and **Masked CatVTON** (`CatVTONPipeline` + SCHP segmentation).

### Hardware & Environment
* **GPU:** NVIDIA GeForce RTX 5060 (Blackwell Architecture, Compute Capability `sm_120`)
* **VRAM:** 8,151 MiB (~8 GB total, ~1.8–2.1 GB allocated during active inference)
* **CUDA Driver Version:** 13.3 (Driver 610.88)
* **PyTorch:** `2.12.0.dev20260408+cu128` (Compiled with native `sm_120` support)
* **Diffusers:** `0.29.2`
* **Transformers:** `4.46.3`
* **Inference Resolution:** 768 × 1024 (Native CatVTON resolution)
* **Diffusion Steps:** 50 steps (DDIM Scheduler, Guidance Scale: 2.5)

---

## 2. Experimental Results & Output Catalog

All runs completed at full **768×1024** resolution without requiring OOM downscaling fallbacks.

| Person | Garment | Garment Type | Category | Pipeline | Output File Path | Resolution | Step Time / Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **demo1** | `full-t-shirt.jpg` | Western Wear | Upper | Mask-Free | `test_images/output/demo1_on_full-t-shirt.png` | 768×1024 | ~2.2s/it (~110s) |
| **demo1** | `shirt.jpg` | Western Wear | Upper | Mask-Free | `test_images/output/demo1_on_shirt.png` | 768×1024 | ~3.1s/it (~155s) |
| **demo1** | `saree_01.png` | Ethnic Wear | Overall | Mask-Free | `test_images/output/demo1_on_saree_01.png` | 768×1024 | ~1.05s/it (~53s) |
| **demo1** | `lehenga_01.png` | Ethnic Wear | Overall | Mask-Free | `test_images/output/demo1_on_lehenga_01.png` | 768×1024 | ~1.05s/it (~53s) |
| **demo1** | `kurti_01.png` | Ethnic Wear | Upper | Mask-Free | `test_images/output/demo1_on_kurti_01.png` | 768×1024 | ~1.06s/it (~54s) |
| **demo2** | `saree_01.png` | Ethnic Wear | Overall | Mask-Free | `test_images/output/demo2_on_saree_01.png` | 768×1024 | ~1.05s/it (~53.8s) |
| **demo2** | `lehenga_01.png` | Ethnic Wear | Overall | Mask-Free | `test_images/output/demo2_on_lehenga_01.png` | 768×1024 | ~1.05s/it (~53.5s) |
| **demo2** | `kurti_01.png` | Ethnic Wear | Upper | Mask-Free | `test_images/output/demo2_on_kurti_01.png` | 768×1024 | ~1.05s/it (~53.2s) |
| **demo1** | `full-t-shirt.jpg` | Western Wear | Upper | Masked | `test_images/output_masked/demo1_on_full-t-shirt.png` | 768×1024 | ~1.05s/it (~53s) |
| **demo1** | `shirt.jpg` | Western Wear | Upper | Masked | `test_images/output_masked/demo1_on_shirt.png` | 768×1024 | ~1.08s/it (~54s) |
| **demo1** | `saree_01.png` | Ethnic Wear | Overall | Masked | `test_images/output_masked/demo1_on_saree_01.png` | 768×1024 | ~1.06s/it (~53.8s) |
| **demo2** | `saree_01.png` | Ethnic Wear | Overall | Masked | `test_images/output_masked/demo2_on_saree_01.png` | 768×1024 | ~1.02s/it (~51.9s) |
| **demo2** | `lehenga_01.png` | Ethnic Wear | Overall | Masked | `test_images/output_masked/demo2_on_lehenga_01.png` | 768×1024 | ~1.06s/it (~53.7s) |
| **demo2** | `kurti_01.png` | Ethnic Wear | Upper | Masked | `test_images/output_masked/demo2_on_kurti_01.png` | 768×1024 | ~1.06s/it (~53.9s) |
| **demo2** | `kurti_01.png` | Ethnic Wear | Upper (Dilated) | Masked | `test_images/output_masked/demo_kurti_dilated.png` | 768×1024 | ~1.05s/it (~53.6s) |

---

## 3. SCHP Human Parsing Diagnostics (`demo2.jpg`)

Self-Correction for Human Parsing (SCHP) using the 18-class ATR schema parsed the model into the following semantic regions:

| Label Index | ATR Label | Pixel Count | % of Image | Role in `overall` Saree / Lehenga Try-On |
| :---: | :--- | :---: | :---: | :--- |
| **00** | **Background** | 549,466 | **82.38%** | 🔒 **Protected (Black / Unchanged)** |
| **02** | **Hair** | 12,942 | **1.94%** | 🔒 **Protected (Black / Unchanged)** |
| **04** | **Upper-clothes** | 39,878 | **5.98%** | ✏️ **Masked (White / Inpainted)** |
| **06** | **Pants** | 44,106 | **6.61%** | ✏️ **Masked (White / Inpainted)** |
| **09** | **Left-shoe** | 3,123 | **0.47%** | 🔒 **Protected (Black / Unchanged)** |
| **10** | **Right-shoe** | 3,055 | **0.46%** | 🔒 **Protected (Black / Unchanged)** |
| **11** | **Face** | 7,085 | **1.06%** | 🔒 **Protected (Black / Unchanged)** |
| **14** | **Left-arm** | 4,026 | **0.60%** | ✏️ **Masked (White / Inpainted)** |
| **15** | **Right-arm** | 3,319 | **0.50%** | ✏️ **Masked (White / Inpainted)** |

* **Total Editable Area:** 91,329 pixels (13.69% of image area)
* **Diagnostic Masks & Maps:**
  * `test_images/output_masked/saree_mask_debug.png` (Raw Inpainting Mask for Saree Try-On)
  * `test_images/output_masked/demo2_schp_color_segmentation.png` (Color-Coded Semantic Parsing Map for Demo 2)
  * `test_images/output_masked/demo1_schp_color_segmentation.png` (Color-Coded Semantic Parsing Map for Demo 1)
  * `test_images/output_masked/kurti_mask_dilated.png` (Downward-Dilated Kurti Inpainting Mask)

---

## 4. Architectural Analysis: Mask-Free vs Masked CatVTON

### Key Findings on Ethnic Wear & Silhouette Variance

1. **Saree Pallu & Lehenga Flare Constraint (Masked Pipeline):**
   * In `demo2.jpg`, the source person is wearing a slim-fit top and tight jeans.
   * Because `Background` (82.38%) is strictly protected to preserve the scene, the masked diffusion inpainting is **strictly confined within the narrow contours of the legs and torso**.
   * Consequently, a **saree pallu** (which naturally drapes diagonally across the shoulder/arm into surrounding space) or a **flared lehenga** cannot paint fabric into background space, forcing the garment into the narrow silhouette of the jeans.
2. **Mask-Free Advantages for Ethnic Drapes:**
   * **Mask-Free CatVTON (`CatVTONPix2PixPipeline`)** operates condition-based image editing without a fixed binary mask. It naturally draws flowing lehenga flares, wide hems, and draped saree pallus beyond the source person's original narrow jeans silhouette.
3. **Kurti Mask Dilation Experiment Result:**
   * **Finding:** Applying vertical downward morphological dilation (+75% area expansion extending into the upper thighs) significantly improved the kurti inpainting length in `test_images/output_masked/demo_kurti_dilated.png`, eliminating the premature hem cutoff seen with standard upper-clothes segmentation masks.

---

## 5. Scripts Added
* `services/run_ethnic_tryon_batch.py`: Automated batch runner with VRAM management, PyTorch memory cleanup, and automatic OOM fallback handling.
* `services/debug_saree_mask.py`: Diagnostic utility for inspecting SCHP segmentation masks and pixel distribution.
* `services/run_kurti_dilation_experiment.py`: Morphological dilation experiment for ethnic garment length extension.
