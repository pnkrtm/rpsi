from ForwardModeling.RockPhysics.Models import simple_model_1

def simple_calculate_test():
    Km = 65
    Gm = 28
    rho_m = 2.71

    Ks = 46
    Gs = 23
    rho_s = 2.43

    Kf = 2.41
    rho_f = 950

    phi = 0.1
    phi_s = 0.1

    simple_model_1(Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m)


if __name__ == '__main__':
    simple_calculate_test()
