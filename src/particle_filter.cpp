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

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 300;
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	//cout<<">>Init:"<<endl;
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	//cout<<">>Prediction:"<<endl;
	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];

		if (fabs(yaw_rate) > 0.001) {
			p.x += (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta)) + dist_x(gen);
			p.y += (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t)) + dist_y(gen);
		}
		else {
			p.x += velocity*cos(p.theta)*delta_t + dist_x(gen);
			p.y += velocity*sin(p.theta)*delta_t + dist_y(gen);
		}
		p.theta += yaw_rate*delta_t + dist_theta(gen);

		particles[i] = p;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//cout<<">>Observation:"<<endl;
	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];
		double weight = 1;
		for(int j  = 0; j < observations.size(); j++) {
			LandmarkObs o = observations[j];
			o.x = p.x + o.x*cos(p.theta) - o.y*sin(p.theta);
			o.y = p.y + o.x*sin(p.theta) + o.y*cos(p.theta);
			double min_distance = 1000;
			int final_id = -1;
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s l = map_landmarks.landmark_list[k];
				double distance = dist(o.x,o.y,l.x_f,l.y_f);
				if (distance > sensor_range)
					continue;
				if (distance < min_distance) {
					min_distance = distance;
					final_id = l.id_i-1;
				}
			}
			if (final_id != -1) {
				Map::single_landmark_s l = map_landmarks.landmark_list[final_id];
				double prob = getProbability(o.x,o.y,l.x_f,l.y_f,std_landmark);
				weight *=prob;
			}
		}
		p.weight = weight;
		particles[i] = p;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	uniform_int_distribution<int> int_dist(0,num_particles-1);
	int index = int_dist(gen);

	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	std::vector<Particle> new_particles;
	double beta = 0.0;

	double max_weight = *max_element(weights.begin(), weights.end());

	uniform_real_distribution<double> real_dist(0,max_weight);

	for (int i = 0; i < num_particles; i++) {
		beta += real_dist(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
	weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
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
