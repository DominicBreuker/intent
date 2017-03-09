# Intent classifier

Run `bin/train.sh` to train a new classifier. Model data will be put to `output` folder.
Training data is taken from folder `data/train/intents.json`.
Put a dev set into `data/dev/intents.json`.

Run `bin/run.sh` to start a server serving model predictions at `localhost:5000/predict`.
Afterwards, you can make predictions like so:

```bash
curl -H "Content-Type: application/json" -X POST -d '{"text":"predicte das du bot"}' http://localhost:5000/predict
```
