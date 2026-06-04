class Roster:

    def __init__(self):
        self.agents = []

    def recruit(self, agent_name: str):
        self.agents.append(agent_name)

    def exist_employee(self, agent_name: str) -> bool:
        names = self.agents + [agent_name]
        names = [name.lower().strip() for name in names]
        names = [name.replace(' ', '').replace('_', '') for name in names]
        agent_name = names[-1]
        if agent_name in names[:-1]:
            return True
        return False

    def print_employees(self):
        names = self.agents
        names = [name.lower().strip() for name in names]
        print(f'Employees: {names}')
