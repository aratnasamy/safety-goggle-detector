# 🦺 PPE Detection System

**Developer:** Amix Eniola  
**File:** `ppe_app.py`  
**Model file:** `bestn.pt`

---

## 📝 Project Description

This project is a **PPE (Personal Protective Equipment) Detection System** built with **Computer Vision**.  
It detects if a person is wearing essential safety gear — **helmet**, **safety goggles**, and **reflective vest** — using a pre-trained YOLOv8 model.

It also:
- Calculates a **safety score** (0%, 25%, 50%, 100%)
- Shows a short **poem** and **emoji mood** based on the score
- Saves results into a log file (`ppe_log.txt`) with **date, score, and detected items**

---

## ⚙️ Features

✅ Detects:
- Helmet  
- Safety Goggles  
- Safety Vest  

✅ Calculates protection percentage:

| PPE Items Detected | Score | Poem | Emoji |
|--------------------|--------|------|--------|
| None | 0% | *“No gear, no grace — danger stares you in the face.”* | 😞 |
| One | 25% | *“One piece won’t shield you from the storm — suit up in full form.”* | 😕 |
| Two | 50% | *“Full armed, fully wise — safety is your greatest prize.”* | 😐 |
| All | 100% | *“Fully armed, fully wide — safety is your greatest prize.”* | 😄 |

✅ Automatically logs all detections in `ppe_log.txt`.

---

## 🧠 Tools & Libraries

- Python 3  
- OpenCV  
- Ultralytics YOLOv8  
- NumPy  

---

## 🚀 How to Run

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   python ppe_app.py
   ```

3. Make sure `bestn.pt` is in the same folder as `ppe_app.py`.

4. The camera will open and start detecting PPE items.  
   Detection results will appear on screen and be logged in `ppe_log.txt`.

---

## 📂 Example Log (`ppe_log.txt`)

```
2025-10-06 | Score: 50% | Helmet: Yes | Goggles: No | Vest: Yes
2025-10-06 | Score: 100% | Helmet: Yes | Goggles: Yes | Vest: Yes
```

---

## 💡 Note

Always wear complete PPE for your safety.  
**“Safety first, every time, every place.”**
