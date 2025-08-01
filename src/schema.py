from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Literal
import logging

logger = logging.getLogger(__name__)

class FlightPredictionInput(BaseModel):
    """
    Input schema for the flight price prediction model.
    """
    # Use model_config for Pydantic V2 configuration
    model_config = ConfigDict(
        populate_by_name=True, # Replaces allow_population_by_field_name
        json_schema_extra={ # Replaces schema_extra
            "example": {
                "airline": "Vistara",
                "source_city": "Delhi",
                "departure_time": "Morning",
                "stops": "one",
                "arrival_time": "Night",
                "destination_city": "Mumbai",
                "class": "Business",
                "duration": 15.83,
                "days_left": 26
            }
        }
    )

    airline: Literal['Vistara', 'Air_India', 'Indigo', 'GO_FIRST', 'AirAsia', 'SpiceJet']
    source_city: Literal['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
    departure_time: Literal['Morning', 'Early_Morning', 'Evening', 'Night', 'Afternoon', 'Late_Night']
    stops: Literal['one', 'zero', 'two_or_more']
    arrival_time: Literal['Night', 'Evening', 'Morning', 'Afternoon', 'Early_Morning', 'Late_Night']
    destination_city: Literal['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
    class_type: Literal['Economy', 'Business'] = Field(alias='class')
    duration: float = Field(gt=0, description="Duration of the flight in hours")
    days_left: int = Field(ge=1, le=50, description="Number of days left for departure (1-50)")

    @model_validator(mode='after')
    def validate_different_cities(self) -> 'FlightPredictionInput':
        if self.source_city == self.destination_city:
            raise ValueError('Source and destination cities must be different')
        return self

    @field_validator('duration')
    def validate_duration_precision(cls, v: float) -> float:
        if v > 48:
             raise ValueError('Duration cannot exceed 48 hours')
        return round(v, 2)

class FlightPredictionOutput(BaseModel):
    """
    Output schema for the prediction response.
    """
    predicted_price: float = Field(gt=0, description="Predicted flight price in currency units")
    
    @field_validator('predicted_price')
    def round_price(cls, v: float) -> float:
        return round(v, 2)
