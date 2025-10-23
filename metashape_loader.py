import xml.etree.ElementTree as ET
import numpy as np
import zipfile

class Sensor:
    def __init__(self):
        self.id = None
        self.label = None
        self.type = None
        self.resolution = {"width": None, "height": None}
        self.properties = {}
        self.bands = []
        self.data_type = None
        self.calibration = {
            "type": None,
            "class": None,
            "resolution": {"width": None, "height": None},
            "f": None,
            "cx": None,
            "cy": None,
            "k1": None,
            "k2": None,
            "k3": None,
            "p1": None,
            "p2": None
        }
        self.covariance = {"params": None, "coeffs": None}
        self.meta = {}
    
    def to_array(self):
        """
        Convert the sensor data to a flat array suitable for uniform blocks in shaders.
        This assumes all values are present.
        """
        # Directly extract and flatten the sensor data
        ret_array = [
            self.resolution["width"],  # Resolution width
            self.resolution["height"],  # Resolution height
            self.properties["pixel_width"],  # Pixel width
            self.properties["pixel_height"],  # Pixel height
            self.properties["focal_length"],  # Focal length
            self.properties["layer_index"],  # Layer index
            self.calibration["f"],  # Focal length calibration
            self.calibration["cx"],  # Camera center x
            self.calibration["cy"],  # Camera center y
            self.calibration["k1"],  # Radial distortion k1
            self.calibration["k2"],  # Radial distortion k2
            self.calibration["k3"],  # Radial distortion k3
            self.calibration["p1"],  # Tangential distortion p1
            self.calibration["p2"],  # Tangential distortion p2
        ]
        
        # Flatten the covariance coefficients (assuming there are 24 values)
        ret_array += self.covariance["coeffs"][:24]  # Limit to 24 values if necessary
        
        return np.array(ret_array, dtype=np.float32)
    
    def __str__(self):
        return f"Sensor(id={self.id}, label={self.label}, type={self.type}, resolution={self.resolution}, " \
               f"properties={self.properties}, bands={self.bands}, data_type={self.data_type}, " \
               f"calibration={self.calibration}, covariance={self.covariance}, meta={self.meta})"

def load_sensors_from_xml(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        with zip_ref.open("doc.xml") as file:
             # Parse the XML file
            tree = ET.parse(file)
            root = tree.getroot()
        
    # Navigate to the sensor element

    sensor_elems = root.findall("./chunks/chunk/sensors/sensor")
    if not sensor_elems:
        raise ValueError("No <sensor> element found in the XML file.")

    sensors = []
    for sensor_elem in sensor_elems:
        # Create a Sensor object
        sensor = Sensor()

        # Fill the sensor fields
        sensor.id = sensor_elem.get("id")
        sensor.label = sensor_elem.get("label")
        sensor.type = sensor_elem.get("type")
        
        # Resolution
        resolution_elem = sensor_elem.find("resolution")
        if resolution_elem is not None:
            sensor.resolution["width"] = int(resolution_elem.get("width"))
            sensor.resolution["height"] = int(resolution_elem.get("height"))
        
        # Properties
        for prop_elem in sensor_elem.findall("property"):
            name = prop_elem.get("name")
            value = prop_elem.get("value")
            if name and value:
                sensor.properties[name] = float(value) if "." in value else int(value)
        
        # Bands
        bands_elem = sensor_elem.find("bands")
        if bands_elem is not None:
            sensor.bands = [band_elem.get("label") for band_elem in bands_elem.findall("band") if band_elem.get("label")]
        
        # Data Type
        data_type_elem = sensor_elem.find("data_type")
        if data_type_elem is not None:
            sensor.data_type = data_type_elem.text
        
        # Calibration
        calibration_elem = sensor_elem.find("calibration")
        if calibration_elem is not None:
            sensor.calibration["type"] = calibration_elem.get("type")
            sensor.calibration["class"] = calibration_elem.get("class")
            
            resolution_elem = calibration_elem.find("resolution")
            if resolution_elem is not None:
                sensor.calibration["resolution"]["width"] = int(resolution_elem.get("width"))
                sensor.calibration["resolution"]["height"] = int(resolution_elem.get("height"))
            
            for field in ["f", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]:
                field_elem = calibration_elem.find(field)
                if field_elem is not None:
                    sensor.calibration[field] = float(field_elem.text)
        
        # Covariance
        covariance_elem = sensor_elem.find("covariance")
        if covariance_elem is not None:
            params_elem = covariance_elem.find("params")
            coeffs_elem = covariance_elem.find("coeffs")
            if params_elem is not None:
                sensor.covariance["params"] = params_elem.text.split()
            if coeffs_elem is not None:
                sensor.covariance["coeffs"] = list(map(float, coeffs_elem.text.split()))
        
        # Meta
        meta_elem = sensor_elem.find("meta")
        if meta_elem is not None:
            for prop_elem in meta_elem.findall("property"):
                name = prop_elem.get("name")
                value = prop_elem.get("value")
                if name and value:
                    sensor.meta[name] = value

        sensors.append(sensor)

    return sensors

def load_sensor_from_xml(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        with zip_ref.open("doc.xml") as file:
             # Parse the XML file
            tree = ET.parse(file)
            root = tree.getroot()
        
    # Navigate to the sensor element
    sensor_elem = root.find("./chunks/chunk/sensors/sensor")
    if sensor_elem is None:
        raise ValueError("No <sensor> element found in the XML file.")
    
    # Create a Sensor object
    sensor = Sensor()
    
    # Fill the sensor fields
    sensor.id = sensor_elem.get("id")
    sensor.label = sensor_elem.get("label")
    sensor.type = sensor_elem.get("type")
    
    # Resolution
    resolution_elem = sensor_elem.find("resolution")
    if resolution_elem is not None:
        sensor.resolution["width"] = int(resolution_elem.get("width"))
        sensor.resolution["height"] = int(resolution_elem.get("height"))
    
    # Properties
    for prop_elem in sensor_elem.findall("property"):
        name = prop_elem.get("name")
        value = prop_elem.get("value")
        if name and value:
            sensor.properties[name] = float(value) if "." in value else int(value)
    
    # Bands
    bands_elem = sensor_elem.find("bands")
    if bands_elem is not None:
        sensor.bands = [band_elem.get("label") for band_elem in bands_elem.findall("band") if band_elem.get("label")]
    
    # Data Type
    data_type_elem = sensor_elem.find("data_type")
    if data_type_elem is not None:
        sensor.data_type = data_type_elem.text
    
    # Calibration
    calibration_elem = sensor_elem.find("calibration")
    if calibration_elem is not None:
        sensor.calibration["type"] = calibration_elem.get("type")
        sensor.calibration["class"] = calibration_elem.get("class")
        
        resolution_elem = calibration_elem.find("resolution")
        if resolution_elem is not None:
            sensor.calibration["resolution"]["width"] = int(resolution_elem.get("width"))
            sensor.calibration["resolution"]["height"] = int(resolution_elem.get("height"))
        
        for field in ["f", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]:
            field_elem = calibration_elem.find(field)
            if field_elem is not None:
                sensor.calibration[field] = float(field_elem.text)
    
    # Covariance
    covariance_elem = sensor_elem.find("covariance")
    if covariance_elem is not None:
        params_elem = covariance_elem.find("params")
        coeffs_elem = covariance_elem.find("coeffs")
        if params_elem is not None:
            sensor.covariance["params"] = params_elem.text.split()
        if coeffs_elem is not None:
            sensor.covariance["coeffs"] = list(map(float, coeffs_elem.text.split()))
    
    # Meta
    meta_elem = sensor_elem.find("meta")
    if meta_elem is not None:
        for prop_elem in meta_elem.findall("property"):
            name = prop_elem.get("name")
            value = prop_elem.get("value")
            if name and value:
                sensor.meta[name] = value
    
    return sensor

class Camera:
    def __init__(self, id, sensor_id, component_id, label, enabled, transform,
                 rotation_covariance, location_covariance, orientation):
        self.id = id
        self.sensor_id = sensor_id
        self.component_id = component_id
        self.label = label
        self.enabled = enabled
        self.transform = transform
        self.rotation_covariance = rotation_covariance
        self.location_covariance = location_covariance
        self.orientation = orientation

    def __repr__(self):
        return (
            f"Camera(id={self.id}, sensor_id={self.sensor_id}, component_id={self.component_id}, "
            f"label='{self.label}', enabled={self.enabled}, transform={self.transform}, "
            f"rotation_covariance={self.rotation_covariance}, location_covariance={self.location_covariance}, "
            f"orientation={self.orientation})"
        )

def load_cameras_from_xml(file_path):
    """
    Load cameras from a Metashape XML file and return chunk component transform details.

    :param filepath: Path to the XML file.
    :return: Tuple containing a list of Camera objects, the rotation matrix (3x3), and the scale value.
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        with zip_ref.open("doc.xml") as file:
    # Parse the XML file
            tree = ET.parse(file)
            root = tree.getroot()

    # Find the first <component> in <chunk/components>
    component_elem = root.find("./chunks/chunk/components/component")
    if component_elem is None:
        return [], None, None  # No component found

    # Extract transform data
    transform_elem = component_elem.find("transform")
    if transform_elem is not None:
        rotation_elem = transform_elem.find("rotation")
        if rotation_elem is not None:
            rotation_matrix = [float(value) for value in rotation_elem.text.split()]
        else:
            rotation_matrix = None

        translation_elem = transform_elem.find("translation")
        if translation_elem is not None:
            translation  = [float(value) for value in translation_elem.text.split()]
        else:
            translation  = None

        scale_elem = transform_elem.find("scale")
        scale_value = float(scale_elem.text.split()[0]) if scale_elem is not None else None
    else:
        rotation_matrix = None
        scale_value = None

    # Find the <cameras> section
    cameras_section = root.find("./chunks/chunk/cameras")
    if cameras_section is None:
        return [], rotation_matrix, scale_value  # No cameras found

    cameras = []

    # Iterate through each <camera> element
    for camera_elem in cameras_section.findall("camera"):
        camera_id = int(camera_elem.get("id")) if camera_elem.get("id") else None
        sensor_id = int(camera_elem.get("sensor_id")) if camera_elem.get("sensor_id") else None
        component_id = int(camera_elem.get("component_id")) if camera_elem.get("component_id") else None
        label = camera_elem.get("label") if camera_elem.get("label") else None
        enabled = camera_elem.get("enabled") == "true" if camera_elem.get("enabled") else None

        if(component_id != None):
            # Parse transform (space-separated string to list of floats)
            transform = [
                float(value) for value in camera_elem.findtext("transform", "").split()
            ]

            # Parse rotation_covariance (if exists)
            rotation_covariance = [
                float(value)
                for value in camera_elem.findtext("rotation_covariance", "").split()
            ]

            # Parse location_covariance (if exists)
            location_covariance = [
                float(value)
                for value in camera_elem.findtext("location_covariance", "").split()
            ]

            # Parse orientation
            orientation = int(camera_elem.findtext("orientation", "0"))

            # Create Camera object
            camera = Camera(
                id=camera_id,
                sensor_id=sensor_id,
                component_id=component_id,
                label=label,
                enabled=enabled,
                transform=transform,
                rotation_covariance=rotation_covariance,
                location_covariance=location_covariance,
                orientation=orientation,
            )
            cameras.append(camera)

    #If rotation or translation or scale are None (or one of them) fallback to identity.
    if rotation_matrix is None:
        rotation_matrix = [1,0,0,0,1,0,0,0,1]
    
    if translation is None:
        translation = [0,0,0]
    
    if scale_value is None:
        scale_value = 1

    return cameras, rotation_matrix, translation, scale_value

