"""Phone reports use the Companion relay; model requests use the existing owner permit path."""
from fastapi import APIRouter, Body, Depends, Request
from pydantic import ValidationError

from colonymind import devices
from colonymind.api.auth import KEY_PRINCIPAL, request_authority, speaks_for_owner
from colonymind.hands.sessions import refuse_hand_session

router = APIRouter(tags=["devices"])
APP = "/v1/host/owner/devices"
MODEL = "/v1/mind/device-invocations"


def companion(request: Request):
    # Same explicit Companion platform and manage grant as the live owner-location route.
    # The adapter's own instance credential cannot report OS grants or change device authorization.
    authority = request_authority(request)
    if not (authority.authenticated and not authority.anonymous and authority.credential_id
            and authority.principal_id != KEY_PRINCIPAL and "companion" in authority.owner_platforms
            and "manage" in authority.access):
        devices.fail(403, "not_companion_owner")


def service():
    from colonymind.api.routers.mind import get_mind
    mind = get_mind()
    if mind is None:
        devices.fail(503, "device_registry_unavailable")
    return devices.Devices(mind)


def parsed(model, body):
    # Validation errors must not echo credential-bearing input into an HTTP response.
    try:
        return model.model_validate(body)
    except (ValidationError, ValueError):
        devices.fail(422, "invalid_device_payload")


app_router = APIRouter(prefix=APP, dependencies=[Depends(companion), Depends(refuse_hand_session)])


@app_router.get("")
async def inventory():
    return {"devices": service().list_devices()}


@app_router.get("/{device_id}")
async def detail(device_id: devices.Id):
    return service().device(device_id)


@app_router.put("/{device_id}")
async def report(device_id: devices.Id, body: object = Body()):
    return service().report(device_id, parsed(devices.DeviceReport, body))


@app_router.get("/{device_id}/capabilities/{capability}")
async def permission(device_id: devices.Id, capability: devices.Capability):
    value = service().device(device_id)
    return {"revision": value["revision"], **devices.Devices._capability(value, capability)}


@app_router.patch("/{device_id}/capabilities/{capability}/authorization")
async def grant(device_id: devices.Id, capability: devices.Capability, body: object = Body()):
    return service().grant(device_id, capability, parsed(devices.GrantEdit, body))


@app_router.get("/{device_id}/invocations")
async def pending(device_id: devices.Id):
    return {"invocations": service().invocations(device_id)}


@app_router.post("/{device_id}/invocations/{invocation_id}/claim")
async def claim(device_id: devices.Id, invocation_id: devices.Id):
    return service().claim(device_id, invocation_id)


@app_router.post("/{device_id}/invocations/{invocation_id}/result")
async def result(device_id: devices.Id, invocation_id: devices.Id, body: object = Body()):
    return service().result(device_id, invocation_id, parsed(devices.Result, body))


@router.post(MODEL, dependencies=[Depends(refuse_hand_session)])
async def invoke(request: Request, body: object = Body()):
    if not speaks_for_owner(request):
        devices.fail(403, "not_owner")
    return service().request(parsed(devices.Invoke, body))


@router.get("/v1/mind/device-capabilities", dependencies=[Depends(refuse_hand_session)])
async def model_inventory(request: Request):
    if not speaks_for_owner(request):
        devices.fail(403, "not_owner")
    return {"devices": service().list_devices()}


@router.get(MODEL + "/{invocation_id}", dependencies=[Depends(refuse_hand_session)])
async def invocation(invocation_id: devices.Id, request: Request):
    if not speaks_for_owner(request):
        devices.fail(403, "not_owner")
    return service().invocation(invocation_id)


router.include_router(app_router)
