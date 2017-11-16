/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"
#include <random>


std::default_random_engine ENGINE;

using namespace std;

double ParticleFilter::gaussian_random(double mean, double deviation) {
    std::normal_distribution<double> dist(mean, 0);
    return dist(ENGINE);
}

double ParticleFilter::gaussian_probability(double mu, double sigma, double x) {
    return exp(-(pow(mu - x, 2) / pow(sigma, 2) / 2.0)) / sqrt(2.0 * M_PI * pow(sigma, 2));
}


double ParticleFilter::distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0));
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 10;
    for (int i = 0; i < num_particles; i++) {
        Particle pa = {
            i,
            gaussian_random(x, std[0]),
            gaussian_random(y, std[1]),
            gaussian_random(theta, std[2]),
            1,
        };
        this->particles.push_back(pa);
    }
    is_initialized = true;

}

void ParticleFilter::prediction(
    double delta_t, double std_pos[], double velocity, double yaw_rate) {
    
    double curve = velocity / yaw_rate;
    
    for (auto &pa : this->particles) {
        double new_x, new_y, new_theta;
        if (abs(yaw_rate) >= 0.0001) {
            new_x = pa.x + curve * (sin(pa.theta + yaw_rate * delta_t) - sin(pa.theta));
            new_y = pa.y + curve * (cos(pa.theta) - cos(pa.theta + yaw_rate * delta_t));
            new_theta = pa.theta + yaw_rate * delta_t;
        } else {
            new_x = pa.x + velocity * delta_t * cos(pa.theta);
            new_y = pa.y + velocity * delta_t * sin(pa.theta);
            new_theta = pa.theta;
        }
        pa.x = gaussian_random(new_x, std_pos[0]);
        pa.y = gaussian_random(new_y, std_pos[1]);
        pa.theta = gaussian_random(new_theta, std_pos[2]);
    }

}


// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. 
//       You can read more about this distribution here: 
//       https://en.wikipedia.org/wiki/Multivariate_normal_distribution
// NOTE: The observations are given in the VEHICLE'S coordinate system. 
//       Your particles are located according to the MAP'S coordinate system. 
//       You will need to transform between the two systems.
//       Keep in mind that this transformation requires both rotation AND 
//       translation (but no scaling).
//   The following is a good resource for the theory:
//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement 
//   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html
void ParticleFilter::updateWeights(
    double sensor_range, 
    double std_landmark[], 
    const std::vector<LandmarkObs> &observations, 
    const Map &map_landmarks) {
    
    double x_std = std_landmark[0];
    double y_std = std_landmark[1];
    for (auto &pa : this->particles) {
        vector<LandmarkObs> observations_world;
        pa.sense_x.clear();
        pa.sense_y.clear();
        pa.associations.clear();
        pa.weight = 1.0;
        for (auto &ob :observations) {
            // transformed from local coordinate to world coordinate
            printf("observation (%f, %f)\n", ob.x, ob.y);
            
            double sense_x = ob.x * cos(pa.theta) - ob.y * sin(pa.theta) + pa.x;
            double sense_y = ob.x * sin(pa.theta) + ob.y * cos(pa.theta) + pa.y;
            LandmarkObs obs_t = {
                -1,
                sense_x,
                sense_y
            };
            printf("transformed (%f, %f)\n", sense_x, sense_y);   
            pa.sense_x.push_back(sense_x);
            pa.sense_y.push_back(sense_y);
            vector<double> distances;
            for (auto &landmark : map_landmarks.landmark_list) {
                double distance = this->distance(pa.x, pa.y, landmark.x_f, landmark.y_f);
                if (distance <= sensor_range)
                    distances.push_back(distance);
            }
            auto result = min_element(begin(distances), end(distances));
            auto lm = map_landmarks.landmark_list[::distance(begin(distances), result)];
            obs_t.id = lm.id_i;
            pa.associations.push_back(lm.id_i);
            printf("Landmark (%f, %f)\n", lm.x_f, lm.y_f);
            printf("Nearests (%f, %f)\n", sense_x, sense_y);
            observations_world.push_back(obs_t);
        }
        double prod = 1.0;
        for (auto &landmark : observations_world) {
            auto map_landmark = map_landmarks.landmark_list[landmark.id];
            prod *= this->gaussian_probability(pa.x, x_std, map_landmark.x_f);
            prod *= this->gaussian_probability(pa.y, y_std, map_landmark.y_f);
        }
        printf("weight %f\n", prod);
        pa.weight = prod;
    }
}

void ParticleFilter::resample() {
    std::random_device rd;
    std::mt19937 gen(rd());
    vector<double> weights;
    for (auto &particle : this->particles)
        weights.push_back(particle.weight);
    std::discrete_distribution<> d(weights.begin(), weights.end());
    vector<Particle> particles_new;
    for(int n=0; n<num_particles; ++n) {
        particles_new.push_back(this->particles[d(gen)]);
    }
    this->particles = particles_new;
    this->num_particles = particles_new.size();
}

Particle ParticleFilter::SetAssociations(
    Particle particle, 
    std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    // particle: the particle to assign each listed association, 
    // and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
     particle.sense_x = sense_x;
     particle.sense_y = sense_y;

     return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
