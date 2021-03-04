from django.core.management.base import BaseCommand
from django.utils import timezone
from detect.Bothavgwrite import create_datastore
#from detect.Bothavgfinal import create_datastore
#from detect.bothavglivealerts import feedlive

class Command(BaseCommand):
    help = 'Displays current time'

    def handle(self, *args, **kwargs):
        create_datastore()
        #feedlive('127.0.0.1')
        time = timezone.now().strftime('%X')
        self.stdout.write("It's now %s" % time)