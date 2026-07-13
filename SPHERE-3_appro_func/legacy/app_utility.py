class Configuration:

    def __init__(self, part=None, config_path='input.ini'):
        self.config_path = config_path
        self.config = None

        self.params = {'part': part, 'part_list': '', 'e': '', 'th': '', 'atm': '', 'q': '', 'start': '', 'end': '',
                       'lcount': ''}
        self.part = self.params['part']
        self.part_list = self.params['part_list']
        self.e = self.params['e']
        self.th = self.params['th']
        self.atm = self.params['atm']
        self.q = self.params['q']

        self.moshits = ''
        self.clout = ''

        self.read_config()
        self.get_parameters()
        self.set_filenames()

    def read_config(self):
        import configparser
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)

    def update_part(self, part='14'):
        self.part = part
        self.set_filenames()

    def update_config(self):
        self.set_parameters()
        self.set_filenames()

    def write_config(self):
        self.update_config()
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

    def get_parameter(self, parameter_name, default_section='DEFAULT'):
        return self.config[default_section][parameter_name]

    def set_parameter(self, parameter_name, parameter_value, default_section='DEFAULT'):
        self.config[default_section][parameter_name] = parameter_value

    def get_parameters(self):
        if self.part is None:
            self.part = self.get_parameter('PRMPAR')
        self.part_list = [item for item in self.get_parameter('PRMPARS').split(', ') if item != '']
        self.e = self.get_parameter('E')
        self.th = self.get_parameter('THETAP')
        self.atm = self.get_parameter('ATMOD')
        self.q = self.get_parameter('QMOD')

    def set_parameters(self):
        self.set_parameter('PRMPAR', self.part)
        self.set_parameter('PRMPARS', ', '.join(self.part_list))
        self.set_parameter('E', self.e)
        self.set_parameter('THETAP', self.th)
        self.set_parameter('ATMOD', self.atm)
        self.set_parameter('QMOD', self.q)

    def set_filenames(self):
        self.moshits = f'Q{self.q}_atm{self.atm.zfill(2)}_{self.part.zfill(4)}_{self.e}PeV_{self.th.zfill(2)}'
