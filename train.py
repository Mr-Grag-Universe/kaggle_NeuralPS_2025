import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .process_data import *

# Типы:
BatchFn = Optional[Callable[[Dict[str, Any]], Tuple[torch.Tensor, torch.Tensor]]]
EvalLossFn = Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    body_batch_fn: BatchFn = None,
    eval_loss_fns: Optional[List[EvalLossFn]] = None,
    eval_loss_weights: Optional[List[float]] = None,
) -> float:
    import tqdm
    """
    Выполняет один epoch обучения.
    eval_loss_fns: список функций loss_fn(output, raw_batch) -> Tensor (скаляр).
    eval_loss_weights: веса для этих функций (по умолчанию все 1.0).
    """
    model.train()
    running_loss = 0.0
    n_samples = 0
    if eval_loss_fns is None:
        eval_loss_fns = []
    if eval_loss_weights is None:
        eval_loss_weights = [1.0] * len(eval_loss_fns)
    assert len(eval_loss_fns) == len(eval_loss_weights)

    for batch in tqdm.tqdm(dataloader):
        # получить inputs/targets (еще не на device)
        if body_batch_fn is not None:
            inputs, targets, det_array = body_batch_fn(batch)
        else:
            inputs = batch['inputs']
            targets = batch['targets']
            b, t, y, l = batch['signal'].shape
            det_array = get_target(batch['signal'])

        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs, det_array=det_array)

        main_loss = criterion(outputs, targets)
        aux_loss = 0.0
        for fn, w in zip(eval_loss_fns, eval_loss_weights):
            l = fn(outputs, batch)
            aux_loss = aux_loss + (w * l)

        total_loss = main_loss + aux_loss
        total_loss.backward()
        optimizer.step()

        bsize = targets.size(0)
        running_loss += total_loss.item() * bsize
        n_samples += bsize

    return running_loss / max(1, n_samples)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    body_batch_fn: BatchFn = None,
    eval_loss_fns: Optional[List[EvalLossFn]] = None,
    eval_loss_weights: Optional[List[float]] = None,
) -> float:
    """
    Оценка модели по валидации. Считает основной loss + те же eval_loss_fns (без градиентов).
    """
    model.eval()
    running = 0.0
    n = 0
    if eval_loss_fns is None:
        eval_loss_fns = []
    if eval_loss_weights is None:
        eval_loss_weights = [1.0] * len(eval_loss_fns)
    assert len(eval_loss_fns) == len(eval_loss_weights)

    with torch.no_grad():
        for batch in dataloader:
            if body_batch_fn is not None:
                inputs, targets, det_array = body_batch_fn(batch)
            else:
                inputs = batch['inputs']
                targets = batch['targets']
                b, t, y, l = batch['signal'].shape
                det_array = get_target(batch['signal'])

            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            outputs = model(inputs, det_array=det_array)
            main_loss = criterion(outputs, targets)
            aux_loss = 0.0
            for fn, w in zip(eval_loss_fns, eval_loss_weights):
                l = fn(outputs, batch)
                aux_loss = aux_loss + (w * l)

            total_loss = main_loss + aux_loss
            running += total_loss.item() * targets.size(0)
            n += targets.size(0)
    return running / max(1, n)


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader_train: torch.utils.data.DataLoader,
    dataloader_val: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    num_epochs: int = 20,
    body_batch_fn: BatchFn = None,
    eval_loss_fns: Optional[List[EvalLossFn]] = None,
    eval_loss_weights: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    plot_backend: str = "matplotlib",
    verbose: bool = True,
) -> Dict[str, Any]:
    history = {"train": [], "val": []}
    best_val = float('inf')
    best_state = None

    if plot_backend == "matplotlib":
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots()
        line_train, = ax.plot([], [], label='train_loss')
        line_val, = ax.plot([], [], label='val_loss')
        ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend()
        fig.canvas.draw(); fig.canvas.flush_events()
    elif plot_backend == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='train_loss'))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='val_loss'))
        fig.update_layout(xaxis_title='epoch', yaxis_title='loss')
    else:
        raise ValueError("plot_backend must be 'matplotlib' or 'plotly'")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, dataloader_train, optimizer, criterion, device,
            body_batch_fn=body_batch_fn,
            eval_loss_fns=eval_loss_fns,
            eval_loss_weights=eval_loss_weights
        )
        history['train'].append(train_loss)

        val_loss = None
        if dataloader_val is not None:
            val_loss = validate(
                model, dataloader_val, criterion, device,
                body_batch_fn=body_batch_fn,
                eval_loss_fns=eval_loss_fns,
                eval_loss_weights=eval_loss_weights
            )
            history['val'].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }
                if save_path is not None:
                    torch.save(best_state, save_path)

        epochs_x = list(range(1, epoch + 1))
        if plot_backend == "matplotlib":
            line_train.set_data(epochs_x, history['train'])
            if history['val']:
                line_val.set_data(epochs_x, history['val'])
            ax.relim(); ax.autoscale_view()
            fig.canvas.draw(); fig.canvas.flush_events()
        else:
            fig.data[0].x = epochs_x; fig.data[0].y = history['train']
            fig.data[1].x = epochs_x; fig.data[1].y = history['val'] if history['val'] else []

        if verbose:
            if val_loss is None:
                print(f"Epoch {epoch}/{num_epochs}  train_loss={train_loss:.6f}  time={time.time()-t0:.1f}s")
            else:
                print(f"Epoch {epoch}/{num_epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  time={time.time()-t0:.1f}s")

    return {"history": history, "best_state": best_state, "best_val": best_val, "fig": fig}



# -------------------------
# Пример использования (шаблон)
# -------------------------
# Определите body_batch_fn, соответствующий вашему коду:
# def my_body_batch_fn(batch):
#     cds_signals = batch['signal']                     # [B,112,16,356]
#     inputs = cds_signals.permute(0, 2, 1, 3)          # пример трансформации
#     targets = batch['target'][:,1:]
#     targets = get_noise(cds_signals, targets)         # если нужно
#     return inputs, targets
#
# Затем вызываете:
# result = fit(model=model,
#              optimizer=optimizer,
#              criterion=nn.MSELoss(),
#              dataloader_train=dataloader_train,
#              dataloader_val=dataloader_val,
#              device=device,
#              num_epochs=20,
#              body_batch_fn=my_body_batch_fn,
#              save_path='best_checkpoint.pth',
#              plot_backend='plotly')   # или 'matplotlib'
#
# best = result['best_state']
