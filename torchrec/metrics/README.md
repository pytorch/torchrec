# Readme
## Metrics Overview
### RecMetricModule
RecMetricModule is the abstract class that encapsulates three metrics types: RecMetric, StateMetric and ThroughputMetric. The main attributes:
1. *rec_tasks: Optional[List[RecTaskInfo]]* : this list provides all the tasks information, such as task name and predictions name in the model output. The task information will be used by RecMetrics to get the right model outputs and compute the corresponding metric.
2. *rec_metrics: Optional[RecMetricList]* : this list stores the RecMetric objects that will be updated/computed when calling update() and compute(). Note that RecMetric is inherited from [TorchMetrics.Metrics](https://torchmetrics.readthedocs.io/en/latest/).
3. *throughput_metric: Optional[ThroughputMetric]* : the ThroughputMetric object to calculate the throughput for trained examples.
4. *state_metrics: Optional[Dict[str, StateMetric]]* : the dictionary to map from state metric name to the StateMetric object.

### RecMetric
The key abstract class to represent a model metric (e.g., NE). There are two main APIs provided by [TorchMetrics.Metric](https://torchmetrics.readthedocs.io/en/latest/), update() and compute().
1. *update()* is the API that should be called when the model outputs from a new batch is generated (e.g., model_output = model.forward()). The intention of update() is to update the local state of the metrics based on the new model outputs.
2. *compute()* is the API that should be called when users require the actual global metrics. The computation is based on the global states, so allgather will be performed to gather the states from all the trainers before the provided compute() is called.

### RecMetricComputation
The abstract class represents the computation (local state update and global state aggregation) of a model metric.

### StaticMetric
This is an interface class to report internal states of a component (e.g., the optimizers). The only method is get_metric().

### ThroughputMetric
Implement the window and lifetime throughput logic.

### Relationship between RecMetric, RecMetricComputation & TorchMetrics.Metrics
**RecMetric vs. RecMetricComputation**
RecMetric contains the task information (RecTaskInfo) and the actual computation object (RecMetricsComputation). It also provides additional features (e.g., metric input validation) and optimizations (e.g., fused update).

**RecMetricComputation vs. TorchMetrics.Metrics**
RecMetricComputation is the subclass of [TorchMetrics.Metrics](https://torchmetrics.readthedocs.io/en/latest/). It adds the logic for window metrics and local metrics.

# Examples
## E2E example for RecMetricModule
Here’s an example for the NE metric and the Throughput metric computation. Suppose the input size is 10 batches, we call update() with example model outputs for 10 times, and then call compute() to get the metrics
```
task = RecTaskInfo(
    name="task_name",
    label_name="label",
    prediction_name="prediction",
    weight_name="weight",
)
metrics_config = MetricsConfig(
    rec_tasks=[task],
    rec_metrics={
        RecMetricEnum.NE: RecMetricDef(
            rec_tasks=[task], window_size=10_000_000
        ),
    },
    throughput_metric=ThroughputDef(),
    state_metrics=[],
)
batch_size = 128
metric_module = generate_metric_module(
    RecMetricModule,
    metrics_config=metrics_config,
    batch_size=batch_size,
    world_size=64,
    my_rank=0,
    state_metrics_mapping={},
    device=torch.device("cpu"),
)
n_batches = 10
for _ in range(n_batches):
    model_output = gen_test_batch(
        label_name=task.label_name,
        prediction_name=task.prediction_name,
        weight_name=task.weight_name,
        batch_size=batch_size,
    )
    metric_module.update(model_output)
metrics_result = metric_module.compute()
```
An example metrics_result can be
```
{'ne-task_name|lifetime_ne': tensor([0.8873], dtype=torch.float64), 'ne-task_name|window_ne': tensor([0.6875], dtype=torch.float64), 'throughput-throughput|total_examples': tensor(1400)}
```

## Detailed explanation for usage of RecMetricModule (with tasks)
Here’s an example for the NE metric, the Optimizer metric and the Throughput metric computation. This example is for MTML, i.e. multi-task multi-label, training with 2 tasks

### Metric initialization
```
task_names = ["t1", "t2"]
tasks = gen_test_tasks(task_names)

class Optimizer(StateMetric):
    def __init__(self) -> None:
        self.get_metrics_call = 0
    def get_metrics(self) -> Dict[str, MetricValue]:
        self.get_metrics_call += 1
        return {"learning_rate": torch.tensor(1.0)}
optimizer = Optimizer()

metrics_config = MetricsConfig(
    rec_tasks=tasks,
    rec_metrics={
        RecMetricEnum.NE: RecMetricDef(
            rec_tasks=tasks, window_size=10_000_000
        ),
    },
    throughput_metric=ThroughputDef(),
    state_metrics=[StateMetricEnum.OPTIMIZERS],
)

metric_module = generate_metric_module(
    RecMetricModule,
    metrics_config=metrics_config,
    batch_size=128,
    world_size=64,
    my_rank=0,
    state_metrics_mapping={StateMetricEnum.OPTIMIZERS: optimizer},
    device=torch.device("cpu"),
)
```

### Metric status update
```
_model_output = [
    gen_test_batch(
        label_name=task.label_name,
        prediction_name=task.prediction_name,
        weight_name=task.weight_name,
        batch_size=128,
    )
    for task in tasks
]
model_output = {k: v for d in _model_output for k, v in d.items()}

metric_module.update(model_output)
```

### Metric calculation
```
metrics_result = metric_module.compute()
```
An example value of metrics_result can be:
```
{'ne-t1|lifetime_ne': tensor([1.7369], dtype=torch.float64), 'ne-t1|window_ne': tensor([1.7369], dtype=torch.float64), 'ne-t2|lifetime_ne': tensor([1.5218], dtype=torch.float64), 'ne-t2|window_ne': tensor([1.5218], dtype=torch.float64), 'throughput-throughput|total_examples': tensor(8192), 'optimizers-optimizers|learning_rate': tensor(1.)}
```

## Checkpoint for RecMetricModule
Metric checkpoint can be supported using the sync(), unsync() and reset() APIs of RecMetricModule. Here’s an example of a metric checkpoint agent. Note we need to implement the is_leader function to decide which trainer is responsible for saving and loading the checkpoint.

### MetricsCheckpointAgent
```
class MetricsCheckpointAgent:
    def __init__(
        self,
        pg: dist.ProcessGroup,
        metric_module: RecMetricModule,
    ) -> None:
        self.pg = pg
        self.metric_module = metric_module

    def save_metrics_checkpoint(self) -> Dict[str, Any]:
        states: Dict[str, Any] = {}
        self.metric_module.sync()
        if self.is_leader(self.pg):
            states.update(self.metric_module.state_dict())
        self.metric_module.unsync()
        # Additional steps to save states are omitted
        return states

    def load_metrics_checkpoint(self, states: Dict[str, Any]) -> None:
        # Additional steps to get states are omitted
        self.metric_module.reset()
        if self.is_leader(self.pg):
            self.metric_module.load_state_dict(states)
```

### Save checkpoint
Metric states for all trainers are collected by allgather, and then the leader trainer will store the global states to the states variable
```
states = metrics_checkpoint_agent.save_metrics_checkpoint()
```

### Load checkpoint
The global states is loaded to the leader trainer, and all the other trainers gets reset
```
metrics_checkpoint_agent.load_metrics_checkpoint(states)
```

## E2E example for RecMetric
RecMetric can also be used directly. This will be applicable if we want to use RecMetric and TorchMetric (e.g., torchMetrics.Accuracy) at the same time, as currently RecMetricModule cannot include TorchMetric directly. We provide the sample metrics including AUC, NE, MSE, etc. Here’s an example for getting started with the NE metric.
```
task_names = ["t1", "t2"]
tasks = gen_test_tasks(task_names)
ne = NEMetric(
    world_size=1,
    my_rank=0,
    batch_size=128,
    tasks=tasks,
    compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    window_size=512,
    fused_update_limit=0,
)
labels, predictions, weights = parse_task_model_outputs(tasks, model_output)
ne.update(
    predictions=predictions,
    labels=labels,
    weights=weights,
)
ne_result = ne.compute()
```
An example value of ne_result can be:
```
{'ne-t1|lifetime_ne': tensor([1.4232], dtype=torch.float64), 'ne-t1|window_ne': tensor([1.4232], dtype=torch.float64), 'ne-t2|lifetime_ne': tensor([1.2736], dtype=torch.float64), 'ne-t2|window_ne': tensor([1.2736], dtype=torch.float64)}
```
