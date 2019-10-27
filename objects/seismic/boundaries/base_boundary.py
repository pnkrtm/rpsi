class BaseBoundary:
    @property
    def wave_direction_1(self):
        """
        First wave's direction on boundary
        :return:
        """
        raise NotImplementedError()

    @property
    def wave_direction_2(self):
        """
        Second wave's direction on boundary
        :return:
        """
        raise NotImplementedError()

    @property
    def wave_type_1(self):
        """
        First wave's type on boundary
        :return:
        """
        raise NotImplementedError()

    @property
    def wave_type_2(self):
        """
        First wave's type on boundary
        :return:
        """
        raise NotImplementedError()
