# Custom modules

Drop custom Python modules in this folder and restart Flask.

To replace a default module, use the same `MODULE_ID`.
To add a new stage, use a unique `MODULE_ID` and an appropriate `MODULE_ORDER`.

Required contract:

```python
MODULE_ID = "m4_retrieval"
MODULE_NAME = "My Retrieval Module"
MODULE_ORDER = 40

def process(context):
    return {"your": "output"}
```
