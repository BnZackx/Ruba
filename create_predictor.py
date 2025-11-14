import pickle
import numpy as np

# Gas constant in J/(mol*K)
R_GAS = 8.314

# Kinetic parameters extracted from the document
KINETIC_PARAMETERS = {
    'Orange (Citrus sinensis)': {
        'C0': 52.3,
        'Ea': 68.4 * 1000, # Convert kJ/mol to J/mol
        'A': 2.34e10
    },
    'Baobab (Adansonia digitata)': {
        'C0': 225.8,
        'Ea': 72.9 * 1000,
        'A': 1.07e11
    },
    'Fluted pumpkin (Telfairia occidentalis)': {
        'C0': 85.2,
        'Ea': 66.8 * 1000,
        'A': 8.45e9
    },
    'Spinach (Amaranthus hybridus)': {
        'C0': 62.4,
        'Ea': 75.2 * 1000,
        'A': 3.89e11
    }
}

class VitaminCPredictor:
    """
    Predicts Vitamin C content (Ct) after thermal processing 
    based on first-order kinetics and the Arrhenius equation.
    """

    def __init__(self, parameters):
        """Initializes the predictor with kinetic parameters."""
        self.parameters = parameters

    def get_rate_constant(self, crop_type, temp_celsius):
        """Calculates the degradation rate constant (k) in min⁻¹."""
        if crop_type not in self.parameters:
            raise ValueError(f"Crop type '{crop_type}' not found.")

        params = self.parameters[crop_type]
        Ea = params['Ea']
        A = params['A']
        
        # Convert Celsius to Kelvin
        T_K = temp_celsius + 273.15
        
        # Arrhenius Equation: k = A * exp(-Ea / (R * T_K))
        k = A * np.exp(-Ea / (R_GAS * T_K))
        return k

    def predict(self, crop_type, temp_celsius, time_min):
        """
        Predicts the final Vitamin C content (Ct) in mg/100g.
        
        Args:
            crop_type (str): Name of the crop (e.g., 'Orange (Citrus sinensis)').
            temp_celsius (float): Processing temperature in Celsius.
            time_min (float): Processing time in minutes.
            
        Returns:
            float: Predicted Vitamin C content (mg/100g).
        """
        if crop_type not in self.parameters:
            raise ValueError(f"Crop type '{crop_type}' not found.")

        params = self.parameters[crop_type]
        C0 = params['C0']
        
        # Calculate the rate constant 'k'
        k = self.get_rate_constant(crop_type, temp_celsius)
        
        # First-order kinetic model: Ct = C0 * exp(-k * t)
        Ct = C0 * np.exp(-k * time_min)
        
        return max(0.0, Ct)

if __name__ == '__main__':
    # 1. Create an instance of the predictor
    predictor_model = VitaminCPredictor(KINETIC_PARAMETERS)

    # 2. Save the instance to the requested .pkl file
    file_name = "vitamin_c_predictor.pkl"
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(predictor_model, f)
        
        print(f"Successfully created and saved the file: {file_name}")

        # --- Verification Test ---
        # Orange at 80°C for 60 min. Document value from Table 4.2 is 17.1 ± 0.5.
        test_crop = "Orange (Citrus sinensis)"
        test_temp = 80
        test_time = 60
        predicted_value = predictor_model.predict(test_crop, test_temp, test_time)
        
        print("\nVerification Test:")
        print(f"Input: {test_crop} at {test_temp}°C for {test_time} min")
        print(f"Predicted Vitamin C: {predicted_value:.2f} mg/100g")
        print(f"Reference Value from Table 4.2: 17.1 ± 0.5 mg/100g ")

    except Exception as e:
        print(f"An error occurred during file creation: {e}")
