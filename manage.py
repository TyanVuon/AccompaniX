#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    # Set the default DJANGO_SETTINGS_MODULE environment variable
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MayBach.settings")

    try:
        # Import Django modules
        from django.core.management import execute_from_command_line
        from django.conf import settings

        # Update settings variables here if needed
        settings.DEBUG = True

        # Execute command line arguments
        execute_from_command_line(sys.argv)
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
