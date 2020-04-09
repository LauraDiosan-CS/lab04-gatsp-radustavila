class UI(object):
    def __init__(self, service):
        self._service = service

    def start(self):
        while True:
            option = int(input('1. Solutie LAB 2\n2. Solutie LAB 4\n3. Exit\n>> '))
            if option == 1:
                print(self._service.lab2())
            if option == 2:
                self._service.lab4()
            if option == 3:
                break


#(39, [1, 3, 2, 9, 7, 4, 8, 6, 5, 1])
#(39, [1, 3, 2, 9, 7, 4, 8, 6, 5, 1])
#(39, [1, 3, 2, 9, 7, 4, 8, 6, 5, 1])