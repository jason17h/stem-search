import os
import boto3
from botocore.client import Config

ACCESS_KEY_ID = os.environ.get('STEMSEARCH_AWS_ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.environ.get('STEMSEARCH_AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.environ.get('STEMSEARCH_AWS_BUCKET_NAME')


# S3 Connect
s3 = boto3.resource(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

listObjSummary = s3.Bucket(BUCKET_NAME).objects.all()

for objSum in listObjSummary:
    print('Item:')
    print(objSum.key)
    objSum.get()['Body'].read()



print ("Done")