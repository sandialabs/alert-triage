import os
import time

class FileLockException(Exception):
    pass

class FileLock(object):

    """ A file lock using the atomic property of symlink(2) on Linux.

    Probably not platform independent.
    """

    def __init__(self, path, timeout=1):
        self.lockfile = os.path.abspath(path) + ".lock"
        self.path = path
        self.pid = os.getpid()
        self.timeout = timeout

        dirname = os.path.dirname(self.lockfile)

        self.id = os.path.join(dirname, "%s.%s" % (self.pid, hash(self.path)))

    def acquire(self):
        """ Acquire the file lock. """
        end_time = time.time()
        if self.timeout is not None and self.timeout:
            end_time += self.timeout
        while True:
            try:
                os.symlink(self.id, self.lockfile)
            except OSError:
                # couldn't create the symlink, do we already have a lock?
                if self.do_i_have_the_lock():
                    # this file lock already owns the lock
                    return
                else:
                    # we already own the lock
                    if self.timeout is not None and time.time() > end_time:
                        raise FileLockException("Could not lock file %s." %
                            self.path)
                    time.sleep(self.timeout is not None \
                        and self.timeout/10 or 0.1)
            else:
                return

    def release(self):
        """ Release the file lock. """
        if not self.is_locked():
            raise FileLockException("File %s is not locked" % self.path)
        elif not self.do_i_have_the_lock():
            raise FileLockException(
                "File %s already locked by a different file lock." % self.path)
        os.unlink(self.lockfile)

    def is_locked(self):
        """ Check if the file is locked. """
        return os.path.islink(self.lockfile)

    def do_i_have_the_lock(self):
        """ Check is this file lock is the one actually locking the file. """
        return os.path.islink(self.lockfile) \
            and os.readlink(self.lockfile) == self.id
