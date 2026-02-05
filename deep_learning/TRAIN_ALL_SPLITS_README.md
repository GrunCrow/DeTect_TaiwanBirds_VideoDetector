# Training Pipeline - All Split Ratios

Este pipeline permite entrenar modelos YOLO automáticamente usando todos los splits generados con diferentes ratios background-target.

## Archivos

- **`train_all_splits.py`**: Script principal en Python
- **`train_all_splits.sh`**: Script bash wrapper para facilitar la ejecución
- Este README

## Estructura de Splits Esperada

El script busca los splits en: `../dataset/csvs/split_ratios/`

```
split_ratios/
├── train/
│   ├── train_95-5.txt
│   ├── train_90-10.txt
│   ├── train_80-20.txt
│   └── ...
├── val/
│   ├── val_95-5.txt
│   ├── val_90-10.txt
│   ├── val_80-20.txt
│   └── ...
└── test/
    └── test.txt
```

## Uso Rápido

### Opción 1: Script Bash (Recomendado)

```bash
# Dar permisos de ejecución
chmod +x train_all_splits.sh

# Entrenar todos los ratios con configuración por defecto
./train_all_splits.sh

# Con Weights & Biases
./train_all_splits.sh --wandb

# Entrenar solo ratios específicos
./train_all_splits.sh --ratios 95-5 80-20 50-50

# Saltar ratios extremos
./train_all_splits.sh --skip-ratios 0-100 100-0

# Configuración personalizada
./train_all_splits.sh --epochs 100 --batch 32 --device 0
```

### Opción 2: Python Directo

```bash
# Entrenar todos los ratios
python train_all_splits.py

# Ver todas las opciones
python train_all_splits.py --help

# Ejemplos
python train_all_splits.py --wandb --epochs 200 --batch 16
python train_all_splits.py --ratios 95-5 80-20 50-50
python train_all_splits.py --skip-ratios 0-100 100-0
```

## Parámetros Principales

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--model` | `yolo26s.pt` | Pesos del modelo pretrained |
| `--epochs` | `200` | Número de épocas de entrenamiento |
| `--batch` | `16` | Tamaño del batch |
| `--device` | `0` | Dispositivo (0 para GPU 0, cpu para CPU) |
| `--patience` | `50` | Paciencia para early stopping |
| `--exp-name` | `yolov26s-splitratios-DeTect1000v1` | Nombre base del experimento |
| `--wandb` | - | Habilitar logging en Weights & Biases |
| `--ratios` | - | Entrenar solo ratios específicos |
| `--skip-ratios` | - | Saltar ratios específicos |

## Funcionamiento

El script automáticamente:

1. **Escanea** el directorio `split_ratios/` buscando pares train-val
2. **Filtra** ratios si se especifican `--ratios` o `--skip-ratios`
3. **Genera** archivos YAML de configuración temporales para cada ratio en `cfg/datasets/auto_generated/`
4. **Entrena** un modelo por cada ratio encontrado
5. **Guarda** resultados en `DeTect-BMMS/runs/{exp_name}_ratio-{XX-YY}/`
6. **Reporta** un resumen final con modelos exitosos y fallidos

## Salida

Cada modelo entrenado genera:

```
DeTect-BMMS/runs/
└── {exp_name}_ratio-{XX-YY}/
    ├── weights/
    │   ├── best.pt      # Mejor modelo
    │   └── last.pt      # Último checkpoint
    ├── results.csv      # Métricas por época
    ├── results.png      # Gráficas de entrenamiento
    └── ...
```

## Configuraciones Temporales

Los archivos YAML generados automáticamente se guardan en:
```
cfg/datasets/auto_generated/
├── DeTect_95-5.yaml
├── DeTect_90-10.yaml
├── DeTect_80-20.yaml
└── ...
```

Estos archivos se pueden revisar para debugging pero no es necesario editarlos manualmente.

## Ejemplos de Uso Común

### Entrenamiento completo con W&B

```bash
./train_all_splits.sh --wandb
```

### Solo entrenar ratios balanceados

```bash
./train_all_splits.sh --ratios 60-40 50-50 40-60
```

### Entrenamiento rápido para pruebas

```bash
./train_all_splits.sh --epochs 50 --patience 20 --skip-ratios 0-100 100-0
```

### Entrenamiento con batch grande y GPU potente

```bash
./train_all_splits.sh --batch 32 --epochs 300
```

### Entrenar solo un ratio específico

```bash
./train_all_splits.sh --ratios 80-20
```

## Monitoreo

Durante el entrenamiento verás:

1. **Lista de ratios detectados** antes de empezar
2. **Progreso actual** (e.g., "Training 3/13")
3. **Información de cada entrenamiento**: experimento, epochs, batch, device
4. **Resumen final** con éxitos y fallos

## Troubleshooting

### Error: "No valid train-val split pairs found"
- Verifica que los splits estén en `../dataset/csvs/split_ratios/train/` y `../dataset/csvs/split_ratios/val/`
- Comprueba que los archivos sigan el formato `train_XX-YY.txt` y `val_XX-YY.txt`

### Error de memoria GPU
- Reduce el `--batch` (prueba con 8 o 4)
- El script limpia automáticamente la cache de CUDA entre entrenamientos

### Saltar entrenamientos fallidos
- El script continúa con el siguiente ratio si uno falla
- Revisa el resumen final para ver qué falló

## Notas

- **Memoria**: El script limpia la cache de GPU entre entrenamientos
- **Tiempo**: Cada entrenamiento puede tomar varias horas dependiendo del hardware
- **Espacio**: Cada modelo genera ~100-500MB de archivos
- **Interrupciones**: Puedes detener con Ctrl+C - los modelos completados se conservan
