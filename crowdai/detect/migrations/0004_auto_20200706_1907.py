# Generated by Django 3.0.6 on 2020-07-06 09:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detect', '0003_auto_20200704_2309'),
    ]

    operations = [
        migrations.AlterField(
            model_name='locationmaster',
            name='locationpassword',
            field=models.TextField(max_length=500, verbose_name='Location Password'),
        ),
    ]
