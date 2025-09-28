import { useEffect, useState } from 'react';
import { Card, Form, Table, Pagination, Row, Col, InputGroup, Button } from 'react-bootstrap';
import { useSearchParams } from 'react-router-dom';
import { useDeployments } from '../contexts/DeploymentContext';

const workloads = [
  "web-frontend", "api-gateway", "user-service", "payment-service",
  "notification-service", "analytics-engine", "database-service"
];

const developers = [
  "John Smith", "Sarah Johnson", "Mike Chen", "Emily Davis",
  "David Wilson", "Lisa Anderson", "Ryan Martinez"
];

const deployers = [
  "DevOps Team", "Release Manager", "Senior Developer",
  "System Administrator", "CI/CD Pipeline"
];

const filterDeployments = (deploymentsList, filters) => {
  let filtered = [...deploymentsList];

  if (filters.developer) {
    filtered = filtered.filter((d) => d.changes_done_by === filters.developer);
  }
  if (filters.deployer) {
    filtered = filtered.filter((d) => d.deployed_by === filters.deployer);
  }
  if (filters.workload) {
    filtered = filtered.filter((d) => d.workload === filters.workload);
  }
  if (filters.date_from) {
    const fromDate = new Date(filters.date_from);
    filtered = filtered.filter((d) => new Date(d.created_at.slice(0, 10)) >= fromDate);
  }
  if (filters.date_to) {
    const toDate = new Date(filters.date_to);
    filtered = filtered.filter((d) => new Date(d.created_at.slice(0, 10)) <= toDate);
  }
  if (filters.search) {
    const searchLower = filters.search.toLowerCase();
    filtered = filtered.filter((d) =>
      Object.values(d).some((value) => value && value.toString().toLowerCase().includes(searchLower))
    );
  }

  return filtered;
};

const sortDeployments = (deploymentsList, sortBy, sortOrder) => {
  const reverse = sortOrder === 'desc';
  if (sortBy === 'created_at') {
    return [...deploymentsList].sort((a, b) => {
      const dateA = new Date(a.created_at);
      const dateB = new Date(b.created_at);
      return reverse ? dateB - dateA : dateA - dateB;
    });
  } else {
    return [...deploymentsList].sort((a, b) => {
      const valA = (a[sortBy] || '').toLowerCase();
      const valB = (b[sortBy] || '').toLowerCase();
      return reverse ? valB.localeCompare(valA) : valA.localeCompare(valB);
    });
  }
};

const DeploymentHistory = () => {
  const { deployments } = useDeployments();
  const [searchParams, setSearchParams] = useSearchParams();

  const getParam = (key, defaultValue) => searchParams.get(key) || defaultValue;

  const [filters, setFilters] = useState({
    search: getParam('search', ''),
    developer: getParam('developer', ''),
    deployer: getParam('deployer', ''),
    workload: getParam('workload', ''),
    date_from: getParam('date_from', ''),
    date_to: getParam('date_to', ''),
  });

  const [sortBy, setSortBy] = useState(getParam('sort_by', 'created_at'));
  const [sortOrder, setSortOrder] = useState(getParam('sort_order', 'desc'));
  const [currentPage, setCurrentPage] = useState(parseInt(getParam('page', 1), 10));
  const [perPage, setPerPage] = useState(parseInt(getParam('per_page', 25), 10));

  useEffect(() => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([k, v]) => { if (v) params.set(k, v); });
    params.set('sort_by', sortBy);
    params.set('sort_order', sortOrder);
    params.set('page', currentPage);
    params.set('per_page', perPage);
    setSearchParams(params);
  }, [filters, sortBy, sortOrder, currentPage, perPage, setSearchParams]);

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters((prev) => ({ ...prev, [name]: value }));
    setCurrentPage(1); // Reset to page 1 on filter change
  };

  const handleSort = (column) => {
    const newOrder = (sortBy === column && sortOrder === 'desc') ? 'asc' : 'desc';
    setSortBy(column);
    setSortOrder(newOrder);
  };

  const filtered = filterDeployments(deployments, filters);
  const sorted = sortDeployments(filtered, sortBy, sortOrder);

  const totalRecords = sorted.length;
  const totalPages = Math.ceil(totalRecords / perPage);
  const startIndex = (currentPage - 1) * perPage;
  const paginated = sorted.slice(startIndex, startIndex + perPage);

  const getSortIcon = (column) => {
    if (sortBy !== column) return 'fa-sort';
    return sortOrder === 'asc' ? 'fa-sort-up' : 'fa-sort-down';
  };

  return (
    <Card className="shadow">
      <Card.Header className="bg-info text-white d-flex justify-content-between align-items-center">
        <h5 className="card-title mb-0 ms-0">
          <i className="fas fa-history me-2"></i>Deployment History
        </h5>
        <Form className="d-flex" onSubmit={(e) => e.preventDefault()}>
          <Form.Control
            type="search"
            name="search"
            placeholder="Search..."
            value={filters.search}
            onChange={handleFilterChange}
            className="me-2"
            style={{ width: '180px' }}
          />
          <Form.Select name="developer" value={filters.developer} onChange={handleFilterChange} className="me-2">
            <option value="">All Developers</option>
            {developers.map((dev) => (
              <option key={dev} value={dev}>{dev}</option>
            ))}
          </Form.Select>
          <Form.Select name="deployer" value={filters.deployer} onChange={handleFilterChange} className="me-2">
            <option value="">All Deployers</option>
            {deployers.map((dep) => (
              <option key={dep} value={dep}>{dep}</option>
            ))}
          </Form.Select>
          <Form.Select name="workload" value={filters.workload} onChange={handleFilterChange} className="me-2">
            <option value="">All Workloads</option>
            {workloads.map((wl) => (
              <option key={wl} value={wl}>{wl}</option>
            ))}
          </Form.Select>
          <Form.Control
            type="date"
            name="date_from"
            value={filters.date_from}
            onChange={handleFilterChange}
            className="me-2"
            title="From"
          />
          <Form.Control
            type="date"
            name="date_to"
            value={filters.date_to}
            onChange={handleFilterChange}
            className="me-2"
            title="To"
          />
          <Button variant="outline-light" type="submit">
            <i className="fas fa-filter"></i>
          </Button>
        </Form>
      </Card.Header>
      <Card.Body className="p-0">
        <div className="table-responsive">
          <Table hover className="align-middle mb-0">
            <thead>
              <tr>
                <th>
                  <a href="#!" onClick={() => handleSort('created_at')}>
                    Created At <i className={`fas ${getSortIcon('created_at')}`}></i>
                  </a>
                </th>
                <th>
                  <a href="#!" onClick={() => handleSort('workload')}>
                    Workload <i className={`fas ${getSortIcon('workload')}`}></i>
                  </a>
                </th>
                <th>Environment</th>
                <th>Version</th>
                <th>Changes</th>
                <th>Config Changes</th>
                <th>
                  <a href="#!" onClick={() => handleSort('changes_done_by')}>
                    Developer <i className={`fas ${getSortIcon('changes_done_by')}`}></i>
                  </a>
                </th>
                <th>
                  <a href="#!" onClick={() => handleSort('deployed_by')}>
                    Deployer <i className={`fas ${getSortIcon('deployed_by')}`}></i>
                  </a>
                </th>
              </tr>
            </thead>
            <tbody>
              {paginated.map((d) => (
                <tr key={d.id}>
                  <td>{d.created_at}</td>
                  <td>{d.workload}</td>
                  <td>{d.environment.charAt(0).toUpperCase() + d.environment.slice(1)}</td>
                  <td>{d.version}</td>
                  <td style={{ whiteSpace: 'pre-line' }}>{d.changes_made}</td>
                  <td style={{ whiteSpace: 'pre-line' }}>{d.config_changes}</td>
                  <td>{d.changes_done_by}</td>
                  <td>{d.deployed_by}</td>
                </tr>
              ))}
              {paginated.length === 0 && (
                <tr>
                  <td colSpan="8" className="text-center text-muted">
                    No deployments found.
                  </td>
                </tr>
              )}
            </tbody>
          </Table>
        </div>
        <nav className="mt-3 mb-2">
          <Pagination className="justify-content-center">
            <Pagination.Prev
              disabled={currentPage === 1}
              onClick={() => setCurrentPage(Math.max(currentPage - 1, 1))}
            />
            {[...Array(totalPages).keys()].map((p) => (
              <Pagination.Item
                key={p + 1}
                active={currentPage === p + 1}
                onClick={() => setCurrentPage(p + 1)}
              >
                {p + 1}
              </Pagination.Item>
            ))}
            <Pagination.Next
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage(Math.min(currentPage + 1, totalPages))}
            />
          </Pagination>
          <div className="d-flex justify-content-center align-items-center mb-0">
            <label className="me-2">Rows:</label>
            <Form.Select
              size="sm"
              className="w-auto"
              value={perPage}
              onChange={(e) => {
                setPerPage(parseInt(e.target.value, 10));
                setCurrentPage(1);
              }}
            >
              <option value="10">10</option>
              <option value="25">25</option>
              <option value="50">50</option>
            </Form.Select>
            <span className="ms-3 small">Total: {totalRecords}</span>
          </div>
        </nav>
      </Card.Body>
    </Card>
  );
};

export default DeploymentHistory;