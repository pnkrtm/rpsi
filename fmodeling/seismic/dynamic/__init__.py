from fmodeling.Utils.classes_keeper import ClassesKeeper

boundary_types = ClassesKeeper()


def register_boundary(K):  # ToDo check names collisions
    boundary_types.add_class(K, None)
    return K