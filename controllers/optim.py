from . import BaseController
import numpy as np
from scipy.optimize import minimize
from collections import namedtuple
from tinyphysics import TinyPhysicsModel, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER, CONTEXT_LENGTH, MAX_ACC_DELTA, CONTROL_START_IDX

PIDPolicy = namedtuple('PIDPolicy', ['p', 'i', 'd'])
State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])

class Controller(BaseController):
    """
    A simple PID controller
    """
    def __init__(self, Kp=0.3, Ki=0.05, Kd=-0.1):
        self.p = Kp
        self.i = Ki
        self.d = Kd
        self.cumulative_integral = 0
        self.prev_error = 0
        self.current_policy = PIDPolicy(p=Kp, i=Ki, d=Kd)
        self.driving_model = TinyPhysicsModel(model_path="./models/tinyphysics.onnx", debug=False)
        self.state_history = []
        self.action_history = []
        self.past_preds = []
        self.prev_optimim_cost = np.inf

    def calc_cost(self, pid_params, target_lataccel, current_lataccel, state, future_plan):
        # Taking the initial guess
        self.current_policy = PIDPolicy(p=pid_params[0], i=pid_params[1], d=pid_params[2])
        N = 10 # Accumulating cost for 10 future states
        prev_error = self.prev_error
        cumulative_error = self.cumulative_integral

        action_history = self.action_history
        state_history = self.state_history
        past_predictions = self.past_preds

        for idx in range(N):
            error = future_plan.lataccel[idx] - current_lataccel
            cumulative_error += error
            error_diff = error - prev_error
            #calculating current action and also using it to predict the next state's lateral accelaration
            action = np.clip(self.current_policy.p * error + self.current_policy.i * cumulative_error + self.current_policy.d * error_diff, STEER_RANGE[0], STEER_RANGE[1])
            action_history.append(action)
            prediction = self.driving_model.get_current_lataccel(
                sim_states=state_history[-CONTEXT_LENGTH:],
                actions=action_history[-CONTEXT_LENGTH:],
                past_preds=past_predictions[-CONTEXT_LENGTH:]
            )
            prediction = np.clip(prediction, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)

            past_predictions.append(prediction)
            state_history.append(State(roll_lataccel=future_plan.roll_lataccel[idx], v_ego=future_plan.v_ego[idx], a_ego=future_plan.a_ego[idx]))
        #Costs calc
        lat_accel_cost = np.mean((np.array(future_plan.lataccel[:N]) - np.array(past_predictions[-N:])) ** 2) * 100
        jerk_cost = np.mean((np.diff(np.array(past_predictions[-N:])) / DEL_T) ** 2) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return total_cost

    def update(self, target_lataccel, current_lataccel, state, future_plan):

        def minimize_wrapper(pid_params):
            '''
            wrapper function that returns total cost computed over a span of 10 future states.
            '''
            return self.calc_cost(pid_params, target_lataccel, current_lataccel, state, future_plan)

        self.state_history.append(state)
        self.past_preds.append(current_lataccel)
        error = (target_lataccel - current_lataccel)
        self.cumulative_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        # We need enough states to start predicting lateral accelaration values. We also want enough future lateral accelaration to calculate
        #steering angles for future states
        if len(self.state_history) > 20 and len(future_plan.lataccel)>=10:
            # Bounds for Kp,Ki and Kd
            bounds = [(0.05, 0.6),
                      (0.02, 0.2),
                      (-0.3, 0.3)]
            #Using baseline as my initial guess
            initial_guess = [self.current_policy.p, self.current_policy.i, self.current_policy.d]
            #SLSQP to minimze cost accumulated over future states
            result = minimize(minimize_wrapper, initial_guess, bounds=bounds, method='slsqp')
            # We don't want to change the policy, if we have a worser cost than before
            if result.fun < self.prev_optimim_cost:
                self.current_policy = PIDPolicy(p=result.x[0], i=result.x[1], d=result.x[2])
                self.prev_optimim_cost = result.fun
            action = self.current_policy.p * error + self.current_policy.i * self.cumulative_integral + self.current_policy.d * error_diff
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            self.action_history.append(action)

        else:
            #Using the current policy if there aren't enough states or future lateral accelarations
            action = self.current_policy.p * error + self.current_policy.i * self.cumulative_integral + self.current_policy.d * error_diff
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            self.action_history.append(action)

        return action
