import argparse


def boolify(s):
    if s == 'True' or s == 'true' or s == 'yes' or s == 'Yes':
        return True
    if s == 'False' or s == 'false' or s == 'no' or s == 'No':
        return False
    raise ValueError("cast error")


def auto_cast(s):
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s


# create a keyvalue class
class KeyValue(argparse.Action):
    # Constructor calling
    def __call__(self, parser, namespace,
                 values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = auto_cast(value)
