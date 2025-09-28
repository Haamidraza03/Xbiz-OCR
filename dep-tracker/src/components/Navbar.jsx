import { Navbar, Nav } from 'react-bootstrap';
import { NavLink } from 'react-router-dom';

const AppNavbar = () => {
  return (
    <Navbar expand="lg" bg="primary" variant="dark" className='fixed-top mb-4 fs-5 shadow-sm'>
      <div className="container">
        <Navbar.Brand as={NavLink} to="/" className='fw-bold fs-4'>
          <i className="fas fa-rocket me-2"></i>Deployment Tracker
        </Navbar.Brand>
        <Navbar.Toggle aria-controls="navbarNav" />
        <Navbar.Collapse id="navbarNav">
          <Nav className="ms-auto">
            <Nav.Link as={NavLink} to="/deployment-form">
              <i className="fas fa-plus-circle me-1"></i>New Deployment
            </Nav.Link>
            <Nav.Link as={NavLink} to="/deployment-history">
              <i className="fas fa-history me-1"></i>Deployment History
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </div>
    </Navbar>
  );
};

export default AppNavbar;