import yaml
import traceback

class Config:
    def __init__(self):
        print("object created")
        try:
            with open('properties.yml','r') as f:
                properties=yaml.load(f,Loader=yaml.FullLoader)
                print(properties)
                self.data_path=properties['settings']['data_path']
                self.data_file=properties['settings']['data_file']
                self.streaming_file=properties['settings']['streaming_file']
                self.kafka_host=properties['settings']['kafka_host']
                self.model_path=properties['model']['model_path']
                self.model_file=properties['model']['model_file']
                self.test_file=properties['settings']['test_file']
        except Exception as e:
            traceback.print_exc()