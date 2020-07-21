from yolo_functions import make_yolov3_model, WeightReader

# You can also save the model just and directly use that
def create_model(weight_path):
    from keras.models import load_model
    model = make_yolov3_model()
    # weight_reader = WeightReader("yolov3.weights")
    weight_reader = WeightReader(weight_path)
    weight_reader.load_weights(model)
    # model.save("model.h5") # If want to save the model then just uncomment this line and line 11 both.
    # model_path = "model.h5"
    return model
