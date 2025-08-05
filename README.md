# IVC Lab Final Codec Optimization

TUM Master Course "Image and Video Compression Lab" in SS25.

## Environment
Python 3.9+

## Main Optimization Structure
 ```│ ...
├─ch4
│      ...
├─ch5
│  │  adaptivequant.py
│  │  deblock.py
│  │  fastmotion.py
│  │  halfpel.py
│  │  main.py
│  │  modedecision.py
│  │  quarterpel.py
│  └─ ...
└─data
    │  ...
 ```

## How to run the codec
### For intra optimization
1. **Deblocking filter**
```bash
python exercises/ch5/deblock.py
```
2. **Adaptive quantization**
```bash
python exercises/ch5/adaptivequant.py
```
### For inter optimization
1. **Block mode decision**
```bash
python exercises/ch5/modedecision.py
```
2. **Fractional-pel motion estimation**
- Half pixel search:
```bash
python exercises/ch5/halfpel.py
```
- Quarter pixel search:
```bash
python exercises/ch5/quarterpel.py
```
3. **Fast mostion estimation**
```bash
python exercises/ch5/fastmotion.py
```
