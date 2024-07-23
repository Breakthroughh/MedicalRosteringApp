from ortools.sat.python import cp_model
import pandas as pd

def main() -> None:
    num_departments = 15
    num_nurses = 100
    num_days = 7
    num_weekday_shifts = 2  # 12-hour shifts
    num_weekend_shifts = 1  # 24-hour shift
    all_departments = range(num_departments)
    all_nurses = range(num_nurses)
    all_days = range(num_days)
    weekdays = range(5)  # Monday to Friday
    weekends = range(5, 7)  # Saturday and Sunday

    # Creates the model.
    model = cp_model.CpModel()

    # Creates shift variables.
    shifts = {}
    for dep in all_departments:
        for n in all_nurses:
            for day in all_days:
                num_shifts = num_weekday_shifts if day in weekdays else num_weekend_shifts
                for s in range(num_shifts):
                    shifts[(dep, n, day, s)] = model.NewBoolVar(f"shift_d{dep}_n{n}_day{day}_s{s}")

    # Each shift in each department is assigned to exactly one nurse.
    for dep in all_departments:
        for day in all_days:
            num_shifts = num_weekday_shifts if day in weekdays else num_weekend_shifts
            model.Add(sum(shifts[(dep, n, day, s)] for n in all_nurses for s in range(num_shifts)) == num_shifts)

    # Allow up to 2 consecutive shifts only during weekdays.
    for n in all_nurses:
        for day in weekdays:
            num_shifts = num_weekday_shifts
            for s in range(num_shifts):
                if s < num_shifts - 1:
                    model.Add(shifts[(dep, n, day, s)] + shifts[(dep, n, day, s + 1)] <= 2)

    # Ensure a 48-hour rest period after a shift or 2 consecutive shifts.
    for n in all_nurses:
        for day in all_days:
            if day < num_days - 2:
                num_shifts = num_weekday_shifts if day in weekdays else num_weekend_shifts
                for s in range(num_shifts):
                    if (dep, n, day, s) in shifts:
                        # 48-hour rest period after a single shift
                        for next_day in range(day + 1, min(day + 3, num_days)):
                            next_shifts = num_weekday_shifts if next_day in weekdays else num_weekend_shifts
                            for next_s in range(next_shifts):
                                if (dep, n, next_day, next_s) in shifts:
                                    model.Add(shifts[(dep, n, day, s)] + shifts[(dep, n, next_day, next_s)] <= 1)
                        
                        # 48-hour rest period after two consecutive shifts
                        if s < num_shifts - 1:
                            for next_day in range(day + 1, min(day + 3, num_days)):
                                next_shifts = num_weekday_shifts if next_day in weekdays else num_weekend_shifts
                                for next_s in range(next_shifts):
                                    if (dep, n, next_day, next_s) in shifts:
                                        model.Add(shifts[(dep, n, day, s)] + shifts[(dep, n, day, s + 1)] + shifts[(dep, n, next_day, next_s)] <= 2)

    # Ensure that nurses do not work in more than one department on the same day.
    for n in all_nurses:
        for day in all_days:
            num_shifts = num_weekday_shifts if day in weekdays else num_weekend_shifts
            model.Add(sum(shifts[(dep, n, day, s)] for dep in all_departments for s in range(num_shifts)) <= 1)

    # Ensure nurses do not work on both weekend days.
    for n in all_nurses:
        model.Add(
            sum(shifts[(dep, n, day, s)] for dep in all_departments for day in weekends for s in range(num_weekend_shifts)) <= 1
        )

    # Try to distribute the shifts evenly.
    total_shifts = num_departments * (num_weekday_shifts * len(weekdays) + num_weekend_shifts * len(weekends))
    min_shifts_per_nurse = total_shifts // num_nurses
    max_shifts_per_nurse = min_shifts_per_nurse + 1 if total_shifts % num_nurses != 0 else min_shifts_per_nurse
    
    for n in all_nurses:
        shifts_worked = []
        for dep in all_departments:
            for day in all_days:
                num_shifts = num_weekday_shifts if day in weekdays else num_weekend_shifts
                shifts_worked.extend(shifts[(dep, n, day, s)] for s in range(num_shifts))
        model.Add(sum(shifts_worked) >= min_shifts_per_nurse)
        model.Add(sum(shifts_worked) <= max_shifts_per_nurse)

    # Creates solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    solver.parameters.enumerate_all_solutions = True

    class NursesPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
        """Print intermediate solutions."""

        def __init__(self, shifts, num_nurses, num_days, num_weekday_shifts, num_weekend_shifts, num_departments, limit):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._shifts = shifts
            self._num_nurses = num_nurses
            self._num_days = num_days
            self._num_weekday_shifts = num_weekday_shifts
            self._num_weekend_shifts = num_weekend_shifts
            self._num_departments = num_departments
            self._solution_count = 0
            self._solution_limit = limit
            self._department_schedule = {dep: {day: {s: None for s in range(num_weekday_shifts if day < 5 else num_weekend_shifts)} for day in range(num_days)} for dep in range(num_departments)}

        def on_solution_callback(self):
            self._solution_count += 1
            for dep in range(self._num_departments):
                for day in range(self._num_days):
                    num_shifts = self._num_weekday_shifts if day < 5 else self._num_weekend_shifts
                    for s in range(num_shifts):
                        for n in range(self._num_nurses):
                            if self.Value(self._shifts[(dep, n, day, s)]):
                                self._department_schedule[dep][day][s] = n

            if self._solution_count >= self._solution_limit:
                self.StopSearch()

        def solutionCount(self):
            return self._solution_count

        def get_department_schedule(self):
            return self._department_schedule

    # Display the first solution.
    solution_limit = 1
    solution_printer = NursesPartialSolutionPrinter(
        shifts, num_nurses, num_days, num_weekday_shifts, num_weekend_shifts, num_departments, solution_limit
    )

    solver.Solve(model, solution_printer)

    department_schedule = solution_printer.get_department_schedule()

    # Display schedule in tabular format
    for dep in range(num_departments):
        print(f"\nDepartment {dep}'s Schedule:")
        schedule_data = []

        for day in range(num_days):
            day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day]
            num_shifts = num_weekday_shifts if day < 5 else num_weekend_shifts

            for s in range(num_shifts):
                nurse = department_schedule[dep][day][s]
                shift_type = f"Shift {s}"
                nurse_str = f"Nurse {nurse}" if nurse is not None else "No nurse assigned"
                schedule_data.append([day_name, shift_type, nurse_str])

        # DataFrame for tabular display
        df_schedule = pd.DataFrame(schedule_data, columns=['Day', 'Shift', 'Nurse'])
        print(df_schedule.to_string(index=False))

if __name__ == "__main__":
    main()
