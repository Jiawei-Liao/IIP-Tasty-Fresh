import docker
import re
import requests
from auth import username, password

# Log data patterns
log_pattern = re.compile(r'time="(?P<time>.*?)" level=(?P<level>\w+) msg="(?P<msg>(?:[^"\\]|\\.)*)"')
startup_pattern = r'Only checking containers which name matches \\"([^\\]+)\\"'
update_pattern = r'Found new ([^:]+):([^ ]+) image \(([^)]+)\)'
ok_pattern = r'Session done'

def listener(container_name='watchtower'):
    try:
        client = docker.from_env()

        container = client.containers.get(container_name)

        for log in container.logs(stream=True, follow=True):
            log_message = log.decode('utf-8').strip()

            # Get data in a json format
            match = log_pattern.match(log_message)
            if match:
                # Create a dictionary from the captured groups
                log_data = match.groupdict()

                if log_data['level'] == 'info':
                    match_startup = re.search(startup_pattern, log_data['msg'])
                    if match_startup:
                        check_container_name = match_startup.group(1)
                        check_container = client.containers.get(check_container_name)
                        image_id = check_container.image.id.split(":")[1][:12]
                        push_db(time=log_data['time'], status='OK', image_id=image_id)
                    
                    match_update = re.search(update_pattern, log_data['msg'])
                    if match_update:
                        image_id = match_update.group(3)
                        push_db(time=log_data['time'], status='UPDATING', image_id=image_id)
                    
                    match_ok = re.search(ok_pattern, log_data['msg'])
                    if match_ok:
                        push_db(time=log_data['time'], status='OK')

    except docker.errors.NotFound:
        print(f"Container '{container_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

ENDPOINT_URL = "https://mobile.tastytrucks.com.au:60052/tastyApiWithResult?endpoint=xpressVision"
AUTH = (username, password)
ADDRESS = "17 Alliance Lane, Clayton, Victoria, 3168"

def push_db(time, status, image_id=None):
    data = {
        'address': ADDRESS,
        'time': time,
        'status': status
    }

    if image_id is not None:
        data['image_id'] = image_id

    try:
        response = requests.post(
            ENDPOINT_URL,
            auth=AUTH,
            json=data,
            headers={'Content-Type': 'application/json'}
        )

        result = response.json()
        if result.get('response') != 'OK':
            print('Error occured while sending status')
    except Exception as e:
        print(e)

if __name__ == '__main__':
    listener()