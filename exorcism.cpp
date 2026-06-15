/*------------------------------------------------------------------------------
| This file is distributed under the BSD 2-Clause License.
| See LICENSE for details.
*-----------------------------------------------------------------------------*/
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "exorcism.h"
#include "cube32.h"

namespace exorcism {

void
exorcism_mngr::pairs_bookmark()
{
	for (auto i = 0u; i < m_pairs.size(); ++i)
		m_pairs_bookmark[i] = m_pairs[i].size();
}

void
exorcism_mngr::pairs_rollback()
{
	for (auto i = 0u; i < m_pairs.size(); ++i)
		m_pairs[i].resize(m_pairs_bookmark[i]);
}

static std::vector<std::array<std::uint32_t, 16>>
make_cube_groups(std::uint32_t dist)
{
	std::array<std::uint32_t, 4> order = {0, 1, 2, 3};
	std::vector<std::array<std::uint32_t, 16>> groups;
	do {
		std::array<std::uint32_t, 16> group = {0};
		for (auto step = 0u; step < dist; ++step) {
			for (auto col = 0u; col < dist; ++col) {
				bool already_changed = false;
				for (auto previous = 0u; previous < step; ++previous)
					already_changed |= order[previous] == col;
				if (order[step] == col)
					group[step * dist + col] = 2;
				else if (already_changed)
					group[step * dist + col] = 1;
				else
					group[step * dist + col] = 0;
			}
		}
		groups.push_back(group);
	} while (std::next_permutation(order.begin(), order.begin() + dist));
	return groups;
}

std::uint32_t
exorcism_mngr::n_cubes()
{
	std::uint32_t n_cubes = 0;
	for (auto &buckt : m_cubes)
		n_cubes += buckt.size();
	return n_cubes;
}

int
exorcism_mngr::add_cube(const cube32 &c, bool add = true)
{
	m_pairs_tmp[0].clear();
	m_pairs_tmp[1].clear();

	const auto n_lits = c.n_lits();
	auto begin = std::max((int)(n_lits - m_max_dist), 0);
	auto end = std::min(m_n_vars, n_lits + m_max_dist);
	for (auto i = begin; i <= end; ++i) {
		for (auto it : m_cubes[i]) {
			const auto dist = distance(c, it);
			if (dist == 1) {
				auto new_cube = merge(c, it);
				m_cubes[i].erase(it);
				return add_cube(new_cube) + 1;
			} else if (dist == 0) {
				m_cubes[i].erase(it);
				m_pairs_tmp[0].clear();
				return 2;
			} else if (dist <= m_max_dist) {
				m_pairs_tmp[dist - 2].push_back(std::make_pair(c, it));
			}
		}
	}
	if (add) {
		m_cubes[n_lits].insert(c);
	}
	for (auto d = 0; d <= (m_max_dist - 2); ++d) {
		std::copy(m_pairs_tmp[d].begin(), m_pairs_tmp[d].end(), std::back_inserter(m_pairs[d]));
	}
	return 0;
}

unsigned
exorcism_mngr::exorlink2()
{
	std::uint32_t n_attempts = 0;
	std::uint32_t n_reshapes = 0;
	std::uint32_t old_size = n_cubes();
	auto &pairs = m_pairs[0];
	auto n_pairs = pairs.size();
	for (auto i = 0; i < n_pairs; ++i) {
		const auto cube_pair = pairs.front();
		pairs.erase(pairs.begin());
		const cube32 cube0 = cube_pair.first;
		const cube32 cube1 = cube_pair.second;
		std::uint32_t cube0_sz = cube0.n_lits();
		std::uint32_t cube1_sz = cube1.n_lits();

		// Remove pair and cubes (cube0, cube1) for now
		auto cube0_it = m_cubes[cube0_sz].find(cube0);
		auto cube1_it = m_cubes[cube1_sz].find(cube1);
		if (cube0_it == m_cubes[cube0_sz].end() || cube1_it == m_cubes[cube1_sz].end())
			continue;
		m_cubes[cube0_sz].erase(cube0_it);
		m_cubes[cube1_sz].erase(cube1_it);

		pairs_bookmark();
		auto n = exorlink(cube0, cube1, 2, &cube_groups2[0]);
		++n_attempts;
		++n_reshapes;
		if (add_cube(n[0], false)) {
			add_cube(n[1]);
		} else if (add_cube(n[1], false)) {
			add_cube(n[0]);
		} else {
			pairs_rollback();
			n = exorlink(cube0, cube1, 2, &cube_groups2[4]);
			if (add_cube(n[0], false)) {
				add_cube(n[1]);
			} else if (add_cube(n[1], false)) {
				add_cube(n[0]);
			} else {
				/* TODO: lit minimization ? */
				m_cubes[cube0_sz].insert(cube0);
				m_cubes[cube1_sz].insert(cube1);
				--n_reshapes;
				pairs_rollback();
				pairs.push_back(cube_pair);
			}
		}
	}
	auto curr_size = n_cubes();
	if (m_verbose) {
		fprintf(stdout, "ExorLink-2");
		fprintf(stdout, ": Que= %5lu", n_pairs);
		fprintf(stdout, "  Att= %4u", n_attempts);
		fprintf(stdout, "  Resh= %4u", n_reshapes);
		fprintf(stdout, "  NoResh= %4d", n_attempts - n_reshapes);
		fprintf(stdout, "  Cubes= %3d", curr_size);
		fprintf(stdout, "  (%d)", old_size - curr_size);
		fprintf(stdout, "\n");
	}
	return old_size - curr_size;
}

unsigned exorcism_mngr::exorlink3()
{
	std::uint32_t n_attempts = 0;
	std::uint32_t n_reshapes = 0;
	std::uint32_t old_size = n_cubes();
	auto &pairs = m_pairs[1];
	auto n_pairs = pairs.size();

	for (auto i = 0; i < n_pairs; ++i) {
		const auto cube_pair = pairs.front();
		pairs.erase(pairs.begin());
		const cube32 cube0 = cube_pair.first;
		const cube32 cube1 = cube_pair.second;
		std::uint32_t cube0_sz = cube0.n_lits();
		std::uint32_t cube1_sz = cube1.n_lits();

		// Remove pair and cubes (cube0, cube1) for now
		auto cube0_it = m_cubes[cube0_sz].find(cube0);
		auto cube1_it = m_cubes[cube1_sz].find(cube1);
		if (cube0_it == m_cubes[cube0_sz].end() || cube1_it == m_cubes[cube1_sz].end())
			continue;
		m_cubes[cube0_sz].erase(cube0_it);
		m_cubes[cube1_sz].erase(cube1_it);

		pairs_bookmark();
		++n_attempts;
		for (auto g = 0u; g < 54u; g += 9u) {
			const auto n = exorlink(cube0, cube1, 3, &cube_groups3[g]);
			for (auto j = 0u; j < 3u; ++j) {
				const auto gain = add_cube(n[j], false);
				if (gain >= 1) {
					for (auto k = 0u; k < 3u; ++k) {
						if (j != k)
							add_cube(n[k]);
					}
					++n_reshapes;
					goto END_LOOP;
				}
				pairs_rollback();
			}
		}
		m_cubes[cube0_sz].insert(cube0);
		m_cubes[cube1_sz].insert(cube1);
END_LOOP: {}
	}
	auto curr_size = n_cubes();
	if (m_verbose) {
		fprintf(stdout, "ExorLink-3");
		fprintf(stdout, ": Que= %5lu", n_pairs);
		fprintf(stdout, "  Att= %4u", n_attempts);
		fprintf(stdout, "  Resh= %4u", n_reshapes);
		fprintf(stdout, "  NoResh= %4d", n_attempts - n_reshapes);
		fprintf(stdout, "  Cubes= %3d", curr_size);
		fprintf(stdout, "  (%d)", old_size - curr_size);
		fprintf(stdout, "\n");
	}
	return old_size - curr_size;
}

unsigned
exorcism_mngr::exorlink4()
{
	std::uint32_t n_attempts = 0;
	std::uint32_t n_reshapes = 0;
	std::uint32_t old_size = n_cubes();
	auto n_pairs = m_pairs[2].size();
	static const auto groups4 = make_cube_groups(4);

	for (auto i = 0u; i < n_pairs; ++i) {
		if (m_pairs[2].empty())
			break;
		const auto cube_pair = m_pairs[2].front();
		m_pairs[2].erase(m_pairs[2].begin());
		const cube32 cube0 = cube_pair.first;
		const cube32 cube1 = cube_pair.second;
		std::uint32_t cube0_sz = cube0.n_lits();
		std::uint32_t cube1_sz = cube1.n_lits();

		auto cube0_it = m_cubes[cube0_sz].find(cube0);
		auto cube1_it = m_cubes[cube1_sz].find(cube1);
		if (cube0_it == m_cubes[cube0_sz].end() || cube1_it == m_cubes[cube1_sz].end())
			continue;

		const auto cubes_before_remove = m_cubes;
		const auto pairs_before_remove = m_pairs;
		m_cubes[cube0_sz].erase(cube0_it);
		m_cubes[cube1_sz].erase(cube1_it);
		const auto cubes_without_pair = m_cubes;
		const auto pairs_without_pair = m_pairs;

		++n_attempts;
		bool accepted = false;
		for (const auto &group : groups4) {
			m_cubes = cubes_without_pair;
			m_pairs = pairs_without_pair;
			const auto n = exorlink(cube0, cube1, 4, const_cast<std::uint32_t *>(group.data()));
			for (auto j = 0u; j < 4u; ++j)
				add_cube(n[j]);
			if (n_cubes() <= old_size) {
				accepted = true;
				++n_reshapes;
				break;
			}
		}

		if (!accepted) {
			m_cubes = cubes_before_remove;
			m_pairs = pairs_before_remove;
			m_pairs[2].push_back(cube_pair);
		}
	}

	auto curr_size = n_cubes();
	if (m_verbose) {
		fprintf(stdout, "ExorLink-4");
		fprintf(stdout, ": Que= %5lu", n_pairs);
		fprintf(stdout, "  Att= %4u", n_attempts);
		fprintf(stdout, "  Resh= %4u", n_reshapes);
		fprintf(stdout, "  NoResh= %4d", n_attempts - n_reshapes);
		fprintf(stdout, "  Cubes= %3d", curr_size);
		fprintf(stdout, "  (%d)", old_size - curr_size);
		fprintf(stdout, "\n");
	}
	return old_size - curr_size;
}

exorcism_mngr::exorcism_mngr(const std::vector<cube32> &original, std::uint32_t n_vars, bool verbose)
	: m_cubes(n_vars + 1),
	  m_n_vars(n_vars),
	  m_max_dist(4),
	  m_pairs(3),
	  m_pairs_tmp(3),
	  m_pairs_bookmark({0, 0, 0, 0}),
	  m_verbose(verbose)
{
	for (auto &pairs : m_pairs)
		pairs.reserve(original.size() * original.size());
	for (const auto &c : original)
		add_cube(c);
}

std::vector<cube32>
exorcism_mngr::run()
{
	std::uint32_t gain = 0;
	std::uint32_t without_improv = 0;
	std::uint32_t iteration = 0;

	do {
		if (m_verbose)
			fprintf(stdout, "\nITERATION: #%2d\n\n", iteration++);
		gain = 0;
		for (auto i = 0u; i < 6u; ++i) {
			gain += exorlink2();
			gain += exorlink3();
			gain += exorlink4();
		}
		if (gain > 0)
			without_improv = 0;
		else
			++without_improv;
	} while (without_improv <= 2);

	if (m_verbose)
		fprintf(stdout, "\n");
	std::vector<cube32> result;
	for (auto buckt : m_cubes)
		for (auto cube : buckt)
			result.push_back(cube);
	return result;
}

}
