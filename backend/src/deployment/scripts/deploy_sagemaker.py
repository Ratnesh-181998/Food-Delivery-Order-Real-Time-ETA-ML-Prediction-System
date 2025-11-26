"""
AWS SageMaker Deployment Script
This script deploys the trained model to SageMaker endpoint
"""

import boto3
import sagemaker
from sagemaker.xgboost import XGBoostModel
from sagemaker import get_execution_role
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerDeployer:
    """
    Class to handle SageMaker model deployment
    """
    
    def __init__(self, region_name='us-east-1'):
        self.region_name = region_name
        self.sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region_name))
        self.s3_client = boto3.client('s3', region_name=region_name)
        
        # Try to get execution role
        try:
            self.role = get_execution_role()
        except:
            # For local development, use a specific role ARN
            self.role = 'arn:aws:iam::ACCOUNT_ID:role/SageMakerRole'
            logger.warning(f"Using default role: {self.role}")
    
    def upload_model_to_s3(self, local_model_path: str, s3_bucket: str, s3_prefix: str = 'models/') -> str:
        """
        Upload trained model to S3
        
        Args:
            local_model_path: Path to local model file
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix/folder
            
        Returns:
            S3 URI of uploaded model
        """
        logger.info(f"Uploading model from {local_model_path} to S3...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{s3_prefix}eta_model_{timestamp}.tar.gz"
        
        # Upload to S3
        self.s3_client.upload_file(local_model_path, s3_bucket, s3_key)
        
        s3_uri = f"s3://{s3_bucket}/{s3_key}"
        logger.info(f"Model uploaded to {s3_uri}")
        
        return s3_uri
    
    def create_model(self, model_data_s3_uri: str, model_name: str = None) -> str:
        """
        Create SageMaker model
        
        Args:
            model_data_s3_uri: S3 URI of model artifacts
            model_name: Name for the model
            
        Returns:
            Model name
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"zomato-eta-model-{timestamp}"
        
        logger.info(f"Creating SageMaker model: {model_name}")
        
        # Create XGBoost model
        xgb_model = XGBoostModel(
            model_data=model_data_s3_uri,
            role=self.role,
            framework_version='1.5-1',
            sagemaker_session=self.sagemaker_session,
            name=model_name
        )
        
        logger.info(f"Model {model_name} created successfully")
        return model_name
    
    def deploy_model(
        self, 
        model_data_s3_uri: str,
        endpoint_name: str = 'zomato-eta-predictor',
        instance_type: str = 'ml.m5.xlarge',
        instance_count: int = 2,
        auto_scaling: bool = True
    ):
        """
        Deploy model to SageMaker endpoint
        
        Args:
            model_data_s3_uri: S3 URI of model artifacts
            endpoint_name: Name for the endpoint
            instance_type: EC2 instance type
            instance_count: Number of instances
            auto_scaling: Enable auto-scaling
            
        Returns:
            Predictor object
        """
        logger.info(f"Deploying model to endpoint: {endpoint_name}")
        
        # Create model
        xgb_model = XGBoostModel(
            model_data=model_data_s3_uri,
            role=self.role,
            framework_version='1.5-1',
            sagemaker_session=self.sagemaker_session
        )
        
        # Deploy model
        predictor = xgb_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            wait=True
        )
        
        logger.info(f"Model deployed successfully to {endpoint_name}")
        
        # Configure auto-scaling if enabled
        if auto_scaling:
            self.configure_auto_scaling(endpoint_name, instance_count)
        
        return predictor
    
    def configure_auto_scaling(
        self, 
        endpoint_name: str,
        min_capacity: int = 2,
        max_capacity: int = 10,
        target_invocations_per_instance: int = 1000
    ):
        """
        Configure auto-scaling for endpoint
        
        Args:
            endpoint_name: Name of the endpoint
            min_capacity: Minimum number of instances
            max_capacity: Maximum number of instances
            target_invocations_per_instance: Target invocations per instance
        """
        logger.info(f"Configuring auto-scaling for {endpoint_name}")
        
        autoscaling_client = boto3.client('application-autoscaling', region_name=self.region_name)
        
        # Register scalable target
        resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
        
        autoscaling_client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity
        )
        
        # Configure scaling policy
        autoscaling_client.put_scaling_policy(
            PolicyName=f"{endpoint_name}-scaling-policy",
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': target_invocations_per_instance,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleInCooldown': 300,
                'ScaleOutCooldown': 60
            }
        )
        
        logger.info("Auto-scaling configured successfully")
    
    def update_endpoint(self, endpoint_name: str, new_model_data_s3_uri: str):
        """
        Update existing endpoint with new model
        
        Args:
            endpoint_name: Name of existing endpoint
            new_model_data_s3_uri: S3 URI of new model
        """
        logger.info(f"Updating endpoint {endpoint_name} with new model")
        
        # Create new model
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_model_name = f"zomato-eta-model-{timestamp}"
        
        xgb_model = XGBoostModel(
            model_data=new_model_data_s3_uri,
            role=self.role,
            framework_version='1.5-1',
            sagemaker_session=self.sagemaker_session,
            name=new_model_name
        )
        
        # Update endpoint
        sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
        
        # Create endpoint config
        endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
        
        xgb_model.create_model()
        
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': new_model_name,
                'InitialInstanceCount': 2,
                'InstanceType': 'ml.m5.xlarge'
            }]
        )
        
        # Update endpoint
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        logger.info(f"Endpoint {endpoint_name} updated successfully")
    
    def delete_endpoint(self, endpoint_name: str):
        """
        Delete SageMaker endpoint
        
        Args:
            endpoint_name: Name of endpoint to delete
        """
        logger.info(f"Deleting endpoint {endpoint_name}")
        
        sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
        
        try:
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting endpoint: {str(e)}")


def main():
    """
    Example deployment workflow
    """
    # Initialize deployer
    deployer = SageMakerDeployer(region_name='us-east-1')
    
    # Configuration
    s3_bucket = 'zomato-ml-models'
    local_model_path = 'models/xgboost_model.tar.gz'
    endpoint_name = 'zomato-eta-predictor'
    
    # Upload model to S3
    model_s3_uri = deployer.upload_model_to_s3(local_model_path, s3_bucket)
    
    # Deploy model
    predictor = deployer.deploy_model(
        model_data_s3_uri=model_s3_uri,
        endpoint_name=endpoint_name,
        instance_type='ml.m5.xlarge',
        instance_count=2,
        auto_scaling=True
    )
    
    logger.info("Deployment complete!")
    
    # Test prediction
    test_data = [[5.2, 12, 1, 0, 1, 0, 20, 0.8, 1.0]]  # Sample features
    prediction = predictor.predict(test_data)
    logger.info(f"Test prediction: {prediction}")


if __name__ == "__main__":
    main()
