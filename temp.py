from app.ml.model_loader import load_model

model = load_model()

print("Production model type:", type(model))
print("Model steps:", getattr(model, "named_steps", "Not a pipeline"))
