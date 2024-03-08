# evalNMT
Automatic evaluation metrics used in natural language generation tasks, including **BLEU**, **ROUGE-L** and **METEOR**.

### Preinstall
Install package `psutil` through pip:

```
pip install psutil
```

### Result Evaluation

Compute the **BLUE-4**, **METEOR** and **ROUGE-L** score by running `eval.py` with the reference file path and the inference result path specified. 

```bash
python3 eval.py --prd_dir data/nngen.codisum.test.msg --gold_dir data/codisum.test.msg 
```

If your predict file or gold file already have index, you can choose option --prd_index and --gold_index instead.


