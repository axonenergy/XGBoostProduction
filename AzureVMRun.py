
from azure.common.credentials import ServicePrincipalCredentials
from azure.mgmt.compute import ComputeManagementClient
import time

def get_credentials(client_id,secret,tenant,subscription_id):
    subscription_id = subscription_id
    credentials = ServicePrincipalCredentials(
        client_id=client_id,
        secret=secret,
        tenant=tenant
    )
    return credentials, subscription_id



def run_script(resource_group_name, vm_name, script_path, client_id,secret,tenant,subscription_id):
    credentials, subscription_id = get_credentials(client_id=client_id,
                                                   secret=secret,
                                                   tenant=tenant,
                                                   subscription_id=subscription_id)

    compute_client = ComputeManagementClient(credentials, subscription_id)

    print('Starting VM')
    async_vm_start = compute_client.virtual_machines.start(resource_group_name, vm_name)
    async_vm_start.wait()
    print('VM Started')

    run_command_parameters = {
        'command_id': 'RunPowerShellScript',
        'script': [script_path]
    }

    # print('Running Script')
    # poller = compute_client.virtual_machines.run_command(
    #     resource_group_name,
    #     vm_name,
    #     run_command_parameters
    # )
    # while not poller.done():
    #     time.sleep(5)
    #
    # result = poller.result()  # Blocking till executed
    # print('TEST OF BALLS')
    # print(result.value[0].message)  # stdout/stderr

    print('Stopping VM')
    async_vm_stop = compute_client.virtual_machines.power_off(resource_group_name, vm_name)
    async_vm_stop.wait()
    print('VM Stopped')


tenant= 'b66eccae-498e-4fbf-b0aa-351a56a6ab65'
client_id= '2f4d40c3-a095-43ca-b664-1c08b96b3c70'
secret= 'hOx6v2IS/rQ:rOjviWABytG:HDWIH44/'


subscription_id= 'a3599178-ccc4-427c-8da2-e88ccc5d40e6'


# Resource Group
GROUP_NAME = 'MSZ_Advanced_Analytics'

# VM
USERNAME = 'szehler'
PASSWORD = 'MSZaa_022020!'
VM_NAME = 'DailyTrdServer'

run_script(resource_group_name=GROUP_NAME,
           vm_name=VM_NAME,
           script_path="C:\\Users\\szehler\\Desktop\\DoDailyPredictionExecutable.bat",
           client_id=client_id,
           secret=secret,
           tenant=tenant,
           subscription_id=subscription_id)