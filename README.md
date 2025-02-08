# Dataset Generation and Model Testing

This repository contains the code for generating three kinds of datasets (ULBD, CDDD, GCDD) and for testing the model trained on ULBD, alongside a comparison with CDDD.

## 1. Dataset Generation

### 1.1 Generate CDDD

**Steps to generate CDDD:**

1. **Get CIF from Materials Project**
   - Run `CDDD_1_get_Cif.py` to obtain the CIF files from the Materials Project.
   - Note: You need to obtain your own API key from the Materials Project website.

2. **Generate Diffraction Patterns**
   - Run `CDDD_2_get_Dataset.py` to generate the diffraction patterns from the CIF files obtained in Step 1.

---

### 1.2 Generate GCDD

**Steps to generate GCDD:**

1. **Generate CIF using Pyxtal**
   - Run `GCDD_1_generate_Cif.py` to create the CIF files.

2. **Generate Diffraction Patterns**
   - Run `GCDD_2_get_Dataset.py` to generate the diffraction patterns from the CIF files obtained in Step 1.

3. **Clean the Dataset**
   - Run `GCDD_3_cleaning_Dataset.py` to remove any NaN and infinite values generated in Step 2.

---

### 1.3 Generate ULBD

**Steps to generate ULBD:**

- Run `ULBD_get_Dataset_*.py` to generate the ULBD for the specific crystal system you are interested in.

---

## 2. Model Testing

### 2.1 Validate Predictions Against Extinction Law

**Steps to validate predictions:**

- Run `test_cubic/tetragonal/trigonal_hexagonal.py` to obtain the top-1 to top-5 accuracies of the model.

---

### 2.2 Test Generalization Ability of Models

**Steps to test generalization ability:**

- Run `test_cubic/tetragonal/trigonal_hexagonal_label_by_extinction.py` to evaluate predicting accuracies.
- Change the test set and model used in the scripts to observe the variations in accuracy.

---

**Note:** Make sure to follow the instructions carefully and adjust any necessary parameters in the scripts as per your requirements.

