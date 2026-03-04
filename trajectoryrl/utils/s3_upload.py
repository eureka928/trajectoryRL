"""Upload a policy pack to S3 for public HTTP access.

Uses boto3 with standard AWS credential chain (env vars, IAM role, etc.).
The pack is serialized with ``json.dumps(pack, sort_keys=True)`` to match
the deterministic hash that ``compute_pack_hash()`` produces.
"""

import json
import logging

logger = logging.getLogger(__name__)


def upload_pack_to_s3(
    pack: dict,
    bucket: str,
    key: str = "pack.json",
    region: str = "us-east-1",
) -> str:
    """Upload a pack dict to S3 and return its public URL.

    Args:
        pack: OPP v1 pack dict.
        bucket: S3 bucket name (must allow public reads via ACL or policy).
        key: Object key (default: ``pack.json``).
        region: AWS region (default: ``us-east-1``).

    Returns:
        Public HTTPS URL of the uploaded object.

    Raises:
        botocore.exceptions.ClientError: On S3 upload failures.
    """
    import boto3  # lazy: not needed by demo mode or validators

    canonical = json.dumps(pack, sort_keys=True)

    s3 = boto3.client("s3", region_name=region)

    logger.info("Uploading pack to s3://%s/%s (%d bytes)", bucket, key, len(canonical))

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=canonical.encode(),
        ContentType="application/json",
        ACL="public-read",
    )

    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    logger.info("Pack uploaded: %s", url)
    return url
