"""Attribution for retained machine assessment quotations."""
VERSION = 'task-artifact-assessment-v1'
ATTRIBUTION = 'host_reported_machine_assessment_unverified'


def quotation_metadata(message):
    """Identify an admitted review bundle even when recall selects an interior chunk."""
    if message.get('role') != 'assistant' or message.get('_task_artifact_assessment') != VERSION:
        return {}
    return {'assessment_context': {
        'kind': VERSION, 'attribution': ATTRIBUTION, 'owner_approval': 'unobserved',
        'interpretation': 'This source bundles a reviewed artifact, a machine assessment and supplied review context. '
            'An excerpt may quote the reviewed artifact rather than the assessment conclusion. '
            'Open the complete source before treating it as a review verdict or recommendation.'}}
