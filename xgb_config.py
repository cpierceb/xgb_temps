# xgb_config.py
import json

class ModelConfig:
    CONFIG_FILE = "current_model_config.json"
    
    @staticmethod
    def set_training_cities(cities_to_include, model_name="default"):
        config = {
            'cities_to_include': cities_to_include,
            'model_name': model_name
        }
        with open(ModelConfig.CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def get_training_cities():
        try:
            with open(ModelConfig.CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get('cities_to_include')
        except:
            return None
    
    @staticmethod
    def get_model_name():
        try:
            with open(ModelConfig.CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get('model_name', 'default')
        except:
            return 'default'